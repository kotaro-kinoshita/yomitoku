import networkx as nx

from .utils.misc import is_right_adjacent, is_bottom_adjacent
from .utils.misc import is_contained, calc_overlap_ratio
from .utils.union_find import UnionFind


def _matching_group_and_cells(nodes):
    """
    グループとヘッダー領域、セル領域のマッチングを行い、セルとグループ、ヘッダーとグループの対応関係を求める
    """

    match = {
        "header_to_group": {},
        "group_to_cells": {},
        "cell_to_groups": {},
        "group_to_header": {},
    }

    if len(nodes["group"]) == 0:
        return match

    B = nx.Graph()
    U = [header.id for header in nodes["header"]]
    V = [group.id for group in nodes["group"]]
    B.add_nodes_from(U, bipartite=0)
    B.add_nodes_from(V, bipartite=1)

    # グループとヘッダー領域の重複率で最大重みマッチング
    for header in nodes["header"]:
        for i, group in enumerate(nodes["group"]):
            ratio = calc_overlap_ratio(header.box, group.box)
            if ratio[0] > 0.0:
                B.add_edge(header.id, group.id, weight=ratio[0])

    kv_items = nx.algorithms.matching.max_weight_matching(
        B, maxcardinality=True, weight="weight"
    )

    matched_header_to_group = {}
    for a, b in kv_items:
        if a in U:
            matched_header_to_group[a] = b
        else:
            matched_header_to_group[b] = a

    # セルとグループのマッチング
    matched_group_to_cells = {}
    matched_cell_to_groups = {}
    for group in nodes["group"]:
        for cell in nodes["cell"] + nodes["empty"]:
            if is_contained(
                group.box,
                cell.box,
                threshold=0.5,
            ):
                matched_group_to_cells.setdefault(group.id, []).append(cell.id)

                if cell.id not in matched_cell_to_groups:
                    matched_cell_to_groups[cell.id] = []

                matched_cell_to_groups[cell.id].append(group.id)

    match["cell_to_groups"] = matched_cell_to_groups
    match["group_to_cells"] = matched_group_to_cells
    match["header_to_group"] = matched_header_to_group
    match["group_to_header"] = {v: k for k, v in matched_header_to_group.items()}

    return match


def _calc_adjacent_header_to_header(dag, match, nodes, group_direction):
    """
    ヘッダーの隣接関係を計算
    グループの方向情報を利用して隣接関係を追加
    例えば、あるグループが水平（H）方向であれば、そのグループに属するヘッダー同士は水平方向の隣接関係を持つ可能性があると判断する
    """

    header_to_group = match["header_to_group"]

    for node in nodes:
        for potential_parent in nodes:
            if node.id == potential_parent.id:
                continue

            potential_parent_group_id = header_to_group.get(potential_parent.id, None)
            node_group_id = header_to_group.get(node.id, None)

            potential_parent_group_direction = group_direction.get(
                potential_parent_group_id, None
            )

            node_group_direction = group_direction.get(node_group_id, None)

            if potential_parent_group_id is None or node_group_id is None:
                continue

            if potential_parent_group_direction == "H" or node_group_direction == "H":
                # 左右の隣接判定
                if is_right_adjacent(potential_parent.box, node.box):
                    dag.add_edge(potential_parent.id, node.id, dir="R")
                    dag.add_edge(node.id, potential_parent.id, dir="L")

            if "V" in potential_parent_group_direction or "V" in node_group_direction:
                # 上下の隣接判定
                if is_bottom_adjacent(potential_parent.box, node.box):
                    dag.add_edge(potential_parent.id, node.id, dir="D")
                    dag.add_edge(node.id, potential_parent.id, dir="U")


def _calc_adjacent_cell_to_cell(dag, match, nodes):
    """
    セルの隣接関係を計算
    セルが同じグリッド領域 or 同じグループに属している場合のみ隣接関係を追加
    """

    cell_to_grid_region = match["cell_to_grid_region"]
    cell_to_groups = match["cell_to_groups"]

    for node in nodes:
        for potential_parent in nodes:
            if node.id == potential_parent.id:
                continue

            # 同じグリッド領域 or 同じグループに属しているか判定
            node_group_id = cell_to_grid_region.get(node.id, None)
            potential_parent_id = cell_to_grid_region.get(potential_parent.id, None)
            if node_group_id is None or potential_parent_id is None:
                node_group_id = cell_to_groups.get(node.id, [])
                potential_parent_id = cell_to_groups.get(potential_parent.id, [])
                if node_group_id is None or potential_parent_id is None:
                    continue
                if set(node_group_id) != set(potential_parent_id):
                    continue
            elif node_group_id != potential_parent_id:
                continue

            # 左右の隣接判定
            if is_right_adjacent(potential_parent.box, node.box):
                dag.add_edge(potential_parent.id, node.id, dir="R")
                dag.add_edge(node.id, potential_parent.id, dir="L")

            # 上下の隣接判定
            if is_bottom_adjacent(potential_parent.box, node.box):
                dag.add_edge(potential_parent.id, node.id, dir="D")
                dag.add_edge(node.id, potential_parent.id, dir="U")


def _calc_adjacent_header_to_cell(dag, match, headers, cells):
    """
    グループ内のヘッダーとセルの隣接関係を計算
    """

    cell_groups = match["cell_to_groups"]
    header_groups = match["header_to_group"]

    for header in headers:
        for cell in cells:
            if is_right_adjacent(header.box, cell.box):
                header_group_id = header_groups.get(header.id, None)
                cell_group_ids = cell_groups.get(cell.id, None)

                if header_group_id is None or cell_group_ids is None:
                    continue

                if header_group_id in cell_group_ids:
                    dag.add_edge(header.id, cell.id, dir="R")
                    dag.add_edge(cell.id, header.id, dir="L")

            if is_bottom_adjacent(header.box, cell.box):
                header_group_id = header_groups.get(header.id, None)
                cell_group_ids = cell_groups.get(cell.id, None)

                if header_group_id is None or cell_group_ids is None:
                    continue

                if header_group_id in cell_group_ids:
                    dag.add_edge(header.id, cell.id, dir="D")
                    dag.add_edge(cell.id, header.id, dir="U")


def _calc_group_direction(dag, match, nodes):
    """
    グループの方向情報を計算
    セルの隣接関係を利用して、グループの方向情報を推定
    """

    group_direction = {}
    group_to_cells = match["group_to_cells"]
    group_to_header = match["group_to_header"]

    for group in nodes["group"]:
        cells = group_to_cells.get(group.id, [])
        is_vertical = False
        for cell_a in cells:
            header_id = group_to_header.get(group.id, None)

            if header_id is not None:
                header_box = dag.nodes[header_id]
                cell_a_box = dag.nodes[cell_a]

                if is_bottom_adjacent(header_box["bbox"], cell_a_box["bbox"]):
                    is_vertical = True
                    group_direction[group.id] = "VHC"
                    break

                if is_right_adjacent(header_box["bbox"], cell_a_box["bbox"]):
                    is_vertical = False
                    group_direction[group.id] = "H"
                    break

            for cell_b in cells:
                if cell_a == cell_b:
                    continue

                cell_a_box = dag.nodes[cell_a]
                cell_b_box = dag.nodes[cell_b]

                if is_bottom_adjacent(
                    cell_b_box["bbox"], cell_a_box["bbox"], rule="hard"
                ):
                    is_vertical = True
                    group_direction[group.id] = "VCC"
                    break

                if is_bottom_adjacent(
                    cell_a_box["bbox"], cell_b_box["bbox"], rule="hard"
                ):
                    group_direction[group.id] = "VCC"
                    is_vertical = True
                    break

        if not is_vertical:
            group_direction[group.id] = "H"

    return group_direction


def _get_grid_regions(match, nodes, group_direction):
    """
    グループの方向情報を利用して、列をマージし、サブテーブルを生成
    """

    cols = [
        group
        for group in nodes["group"]
        if group.id in group_direction and group_direction[group.id] == "VHC"
    ]

    if len(cols) == 0:
        cols = [
            group
            for group in nodes["group"]
            if group.id in group_direction and group_direction[group.id] == "VCC"
        ]

    group_cells = match["group_to_cells"]
    union_find = UnionFind(len(cols))
    for i, col_a in enumerate(cols):
        if col_a.id not in group_cells:
            continue

        for j, col_b in enumerate(cols):
            if col_a.id == col_b.id:
                continue

            if is_right_adjacent(col_a.box, col_b.box, rule="soft"):
                union_find.union(i, j)

            if is_contained(
                col_a.box,
                col_b.box,
                threshold=0.8,
            ) or is_contained(
                col_b.box,
                col_a.box,
                threshold=0.8,
            ):
                union_find.union(i, j)

    grid_regions = []
    for groups in union_find.groups():
        x1 = min([cols[i].box[0] for i in groups])
        y1 = min([cols[i].box[1] for i in groups])
        x2 = max([cols[i].box[2] for i in groups])
        y2 = max([cols[i].box[3] for i in groups])

        grid_regions.append(
            {
                "id": len(grid_regions) + 1,
                "box": [x1, y1, x2, y2],
                "member_ids": [cols[i].id for i in groups],
            }
        )

    return grid_regions


def _matching_cells_within_grid_regions(match, grid_regions):
    """
    グリッド領域に含まれるセルを求める
    """

    matched_cell_and_grid_region = {}
    group_to_cells = match["group_to_cells"]

    for grid_region in grid_regions:
        member_group_ids = grid_region["member_ids"]

        for group_id in member_group_ids:
            if group_id not in group_to_cells:
                continue

            for cell_id in group_to_cells[group_id]:
                matched_cell_and_grid_region[cell_id] = grid_region["id"]

    match["cell_to_grid_region"] = matched_cell_and_grid_region


def calc_cell_relation_dag(nodes):
    match = _matching_group_and_cells(nodes)
    cell_relation_dag = nx.DiGraph()
    for node in nodes["header"] + nodes["cell"] + nodes["empty"]:
        cell_relation_dag.add_node(
            node.id,
            id=node.id,
            bbox=node.box,
            role=node.role,
            contents=node.contents,
        )

    group_direction = _calc_group_direction(cell_relation_dag, match, nodes)

    _calc_adjacent_header_to_cell(
        cell_relation_dag,
        match,
        nodes["header"],
        nodes["cell"] + nodes["empty"],
    )

    _calc_adjacent_header_to_header(
        cell_relation_dag, match, nodes["header"], group_direction
    )

    grid_regions = _get_grid_regions(match, nodes, group_direction)
    _matching_cells_within_grid_regions(match, grid_regions)
    _calc_adjacent_cell_to_cell(
        cell_relation_dag, match, nodes["cell"] + nodes["empty"]
    )

    return cell_relation_dag, match, group_direction, grid_regions
