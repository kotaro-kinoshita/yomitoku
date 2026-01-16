import networkx as nx

from .utils.misc import (
    get_line_with_head,
    calc_overlap_ratio,
    is_contained,
    is_right_adjacent,
    is_bottom_adjacent,
)


from .schemas.table_semantic_parser import KvItemSchema


def _matching_group_and_cells(nodes, groups):
    """
    グループとヘッダー領域、セル領域のマッチングを行い、セルとグループ、ヘッダーとグループの対応関係を求める
    """

    match = {
        "header_to_group": {},
        "group_to_cells": {},
        "cell_to_groups": {},
        "group_to_header": {},
    }

    if len(groups) == 0:
        return match

    B = nx.Graph()
    U = [header.id for header in nodes["header"]]
    V = [group.id for group in groups]
    B.add_nodes_from(U, bipartite=0)
    B.add_nodes_from(V, bipartite=1)

    # グループとヘッダー領域の重複率で最大重みマッチング
    for header in nodes["header"]:
        for i, group in enumerate(groups):
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
    for group in groups:
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

    cell_to_groups = match["cell_to_groups"]

    for node in nodes:
        for potential_parent in nodes:
            if node.id == potential_parent.id:
                continue

            # 同じグリッド領域 or 同じグループに属しているか判定

            node_group_id = cell_to_groups.get(node.id, [])
            potential_parent_id = cell_to_groups.get(potential_parent.id, [])

            if node_group_id is None or potential_parent_id is None:
                continue

            if set(node_group_id) != set(potential_parent_id):
                continue

            # 左右の隣接判定
            if is_right_adjacent(potential_parent.box, node.box):
                dag.add_edge(potential_parent.id, node.id, dir="R")
                dag.add_edge(node.id, potential_parent.id, dir="L")

            # 上下の隣接判定
            if is_bottom_adjacent(potential_parent.box, node.box):
                dag.add_edge(potential_parent.id, node.id, dir="D")
                dag.add_edge(node.id, potential_parent.id, dir="U")


def _calc_group_direction(dag, match, groups):
    """
    グループの方向情報を計算
    セルの隣接関係を利用して、グループの方向情報を推定
    """

    group_direction = {}
    group_to_cells = match["group_to_cells"]
    group_to_header = match["group_to_header"]

    for group in groups:
        cells = group_to_cells.get(group.id, [])
        is_vertical = False
        for cell_a in cells:
            header_id = group_to_header.get(group.id, None)

            if header_id is not None:
                header_box = dag.nodes[header_id]
                cell_a_box = dag.nodes[cell_a]

                if is_bottom_adjacent(header_box["bbox"], cell_a_box["bbox"]):
                    is_vertical = True
                    group_direction[group.id] = "V"
                    break

                if is_right_adjacent(header_box["bbox"], cell_a_box["bbox"]):
                    is_vertical = False
                    group_direction[group.id] = "H"
                    break

        if not is_vertical:
            group_direction[group.id] = "H"

    return group_direction


def get_kv_items_dag(nodes, groups):
    match = _matching_group_and_cells(nodes, groups)
    dag = nx.DiGraph()

    for node in nodes["header"] + nodes["cell"]:
        dag.add_node(
            node.id,
            id=node.id,
            bbox=node.box,
            role=node.role,
            contents=node.contents,
        )

    group_direction = _calc_group_direction(dag, match, groups)

    _calc_adjacent_header_to_cell(
        dag,
        match,
        nodes["header"],
        nodes["cell"],
    )

    _calc_adjacent_header_to_cell(
        dag,
        match,
        nodes["header"],
        nodes["cell"],
    )

    _calc_adjacent_header_to_header(dag, match, nodes["header"], group_direction)

    _calc_adjacent_cell_to_cell(dag, match, nodes["cell"])

    return dag


def parse_kv_items(clustered_nodes, nodes):
    dag = get_kv_items_dag(clustered_nodes, nodes["group"])

    kv_items = []
    for cell in clustered_nodes["cell"]:
        kv_items_row = get_line_with_head(dag, cell.id, dir_value="L")
        kv_items_col = get_line_with_head(dag, cell.id, dir_value="U")

        headers = []
        headers.extend(
            sorted(
                [
                    dag.nodes[h]
                    for h in kv_items_row
                    if dag.nodes[h]["role"] == "header"
                ],
                key=lambda n: n["bbox"][0],
            )
        )

        headers.extend(
            sorted(
                [
                    dag.nodes[h]
                    for h in kv_items_col
                    if dag.nodes[h]["role"] == "header"
                ],
                key=lambda n: n["bbox"][1],
            )
        )

        kv_items.append(
            KvItemSchema(
                id=None,
                key=[header["id"] for header in headers],
                value=cell.id,
            )
        )

    return kv_items, dag
