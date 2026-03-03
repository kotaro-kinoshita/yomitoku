import networkx as nx

from .schemas.table_semantic_parser import KvItemSchema
from .utils.misc import (
    calc_overlap_ratio,
    is_bottom_adjacent,
    is_contained,
    is_right_adjacent,
)

PSEUDO_GROUP_ID = "__unmatched__"


def _matching_group_and_cells(nodes, groups):
    """
    グループとヘッダー領域、セル領域のマッチングを行い、セルとグループ、ヘッダーとグループの対応関係を求める
    ヘッダー・セルともにグループに対してN:1（各ノードは最も重複率の高い1グループに属する）
    どのグループにも属さないノードは擬似グループにまとめる
    """

    match = {
        "header_to_group": {},
        "group_to_cells": {},
        "cell_to_group": {},
        "group_to_headers": {},
    }

    if len(groups) == 0:
        return match

    # セルとグループのマッチング（各セルを最も重複率の高い1グループに割り当て）
    matched_cell_to_group = {}
    for cell in nodes["cell"] + nodes["empty"]:
        best_group_id = None
        best_ratio = 0.0
        for group in groups:
            if is_contained(group.box, cell.box, threshold=0.2):
                ratio = calc_overlap_ratio(cell.box, group.box)[0]
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_group_id = group.id
        if best_group_id is not None:
            matched_cell_to_group[cell.id] = best_group_id

    # ヘッダーとグループのマッチング（N:1、各ヘッダーを最も重複率の高いグループに割り当て）
    matched_header_to_group = {}
    for header in nodes["header"]:
        best_group_id = None
        best_ratio = 0.0
        for group in groups:
            if is_contained(group.box, header.box, threshold=0.2):
                ratio = calc_overlap_ratio(header.box, group.box)[0]
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_group_id = group.id
        if best_group_id is not None:
            matched_header_to_group[header.id] = best_group_id

    # どのグループにも属さないheader/cellを擬似グループにまとめる
    unmatched_cells = [
        c.id
        for c in nodes["cell"] + nodes["empty"]
        if c.id not in matched_cell_to_group
    ]
    unmatched_headers = [
        h.id for h in nodes["header"] if h.id not in matched_header_to_group
    ]

    if unmatched_cells or unmatched_headers:
        pseudo_group_id = PSEUDO_GROUP_ID
        for cell_id in unmatched_cells:
            matched_cell_to_group[cell_id] = pseudo_group_id
        for header_id in unmatched_headers:
            matched_header_to_group[header_id] = pseudo_group_id

    match["cell_to_group"] = matched_cell_to_group
    match["header_to_group"] = matched_header_to_group

    # group_to_cells: グループ→セルリスト（逆引き）
    matched_group_to_cells = {}
    for cell_id, group_id in matched_cell_to_group.items():
        matched_group_to_cells.setdefault(group_id, []).append(cell_id)
    match["group_to_cells"] = matched_group_to_cells

    # group_to_headers: グループ→ヘッダーリスト（N:1の逆引き）
    group_to_headers = {}
    for header_id, group_id in matched_header_to_group.items():
        group_to_headers.setdefault(group_id, []).append(header_id)
    match["group_to_headers"] = group_to_headers

    return match


def _calc_adjacent_header_to_cell(dag, match, headers, cells):
    """
    グループ内のヘッダーとセルの隣接関係を計算
    """

    cell_to_group = match["cell_to_group"]
    header_to_group = match["header_to_group"]

    for header in headers:
        header_group_id = header_to_group.get(header.id, None)
        if header_group_id is None:
            continue

        for cell in cells:
            cell_group_id = cell_to_group.get(cell.id, None)
            if cell_group_id is None:
                continue

            if header_group_id != cell_group_id:
                continue

            if is_right_adjacent(header.box, cell.box):
                dag.add_edge(header.id, cell.id, dir="R")
                dag.add_edge(cell.id, header.id, dir="L")

            if is_bottom_adjacent(header.box, cell.box):
                dag.add_edge(header.id, cell.id, dir="D")
                dag.add_edge(cell.id, header.id, dir="U")


def _calc_adjacent_header_to_header(dag, match, nodes):
    """
    ヘッダーの隣接関係を計算
    同じグループに属するヘッダー同士、またはどちらかが未割り当て（擬似グループ）の場合にエッジを追加
    """

    header_to_group = match["header_to_group"]

    for node in nodes:
        for potential_parent in nodes:
            if node.id == potential_parent.id:
                continue

            potential_parent_group_id = header_to_group.get(potential_parent.id, None)
            node_group_id = header_to_group.get(node.id, None)

            if potential_parent_group_id is None or node_group_id is None:
                continue

            # 同じグループ or どちらかが未割り当ての場合のみ
            is_same_group = potential_parent_group_id == node_group_id
            has_unmatched = (
                potential_parent_group_id == PSEUDO_GROUP_ID
                or node_group_id == PSEUDO_GROUP_ID
            )
            if not is_same_group and not has_unmatched:
                continue

            # 左右の隣接判定
            if is_right_adjacent(potential_parent.box, node.box):
                dag.add_edge(potential_parent.id, node.id, dir="R")
                dag.add_edge(node.id, potential_parent.id, dir="L")

            # 上下の隣接判定
            if is_bottom_adjacent(potential_parent.box, node.box):
                dag.add_edge(potential_parent.id, node.id, dir="D")
                dag.add_edge(node.id, potential_parent.id, dir="U")


def _calc_adjacent_cell_to_cell(dag, match, nodes):
    """
    セルの隣接関係を計算
    セルが同じグリッド領域 or 同じグループに属している場合のみ隣接関係を追加
    """

    cell_to_group = match["cell_to_group"]

    for node in nodes:
        node_group_id = cell_to_group.get(node.id, None)
        if node_group_id is None:
            continue

        for potential_parent in nodes:
            if node.id == potential_parent.id:
                continue

            # 同じグループに属しているか判定
            potential_parent_group_id = cell_to_group.get(potential_parent.id, None)
            if potential_parent_group_id is None:
                continue

            if node_group_id != potential_parent_group_id:
                continue

            # 左右の隣接判定
            if is_right_adjacent(potential_parent.box, node.box):
                dag.add_edge(potential_parent.id, node.id, dir="R")
                dag.add_edge(node.id, potential_parent.id, dir="L")

            # 上下の隣接判定
            if is_bottom_adjacent(potential_parent.box, node.box):
                dag.add_edge(potential_parent.id, node.id, dir="D")
                dag.add_edge(node.id, potential_parent.id, dir="U")


def get_kv_items_dag(nodes, groups):
    match = _matching_group_and_cells(nodes, groups)
    dag = nx.DiGraph()

    for node in nodes["header"] + nodes["cell"] + nodes["empty"]:
        dag.add_node(
            node.id,
            id=node.id,
            bbox=node.box,
            role=node.role,
            contents=node.contents,
        )

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
        nodes["empty"],
    )

    _calc_adjacent_header_to_header(dag, match, nodes["header"])
    _calc_adjacent_cell_to_cell(dag, match, nodes["cell"])

    return dag


def _merge_bbox(box1, box2):
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])

    return [x1, y1, x2, y2]


def _find_root_headers(dag, direction, node_set=None):
    """DAG内のルートヘッダーを検出する。
    指定方向のエッジを他のヘッダーから受けないヘッダーがルート。
    node_setが指定された場合、そのノード集合内のみで判定する。
    """
    if node_set is None:
        node_set = set(dag.nodes)
    headers = [n for n in node_set if dag.nodes[n]["role"] == "header"]
    roots = []
    for h in headers:
        has_parent_header = False
        for u in dag.predecessors(h):
            if u not in node_set:
                continue
            if dag.nodes[u]["role"] != "header":
                continue
            if dag[u][h].get("dir") == direction:
                has_parent_header = True
                break
        if not has_parent_header:
            roots.append(h)
    return roots


def _dfs_collect_kv(dag, node_id, key_path, kv_items, cells, kv_cells, allowed_dir):
    """ルートヘッダーからDFSし、葉(cell/empty)到達時にKvItemを生成する。"""
    node = dag.nodes[node_id]

    if node["role"] in ("cell", "empty"):
        keys = list(key_path)
        box = (
            _merge_bbox(cells[node_id].box, cells[keys[0]].box)
            if keys
            else cells[node_id].box
        )
        kv_items.append(KvItemSchema(id=None, key=keys, value=node_id, box=box))
        kv_cells[node_id] = cells[node_id]
        for k in keys:
            kv_cells[k] = cells[k]
        return

    # headerノード: key_pathに追加して指定方向の子を探索
    new_key_path = key_path + [node_id]
    for v in dag.successors(node_id):
        if v in new_key_path:
            continue
        if dag[node_id][v].get("dir") == allowed_dir:
            _dfs_collect_kv(
                dag, v, new_key_path, kv_items, cells, kv_cells, allowed_dir
            )


def parse_kv_items(clustered_nodes, nodes, cells):
    dag = get_kv_items_dag(clustered_nodes, nodes["group"])

    kv_items = []
    kv_cells = {}

    # 連結クラスタごとに向き推定とDFSを実行
    for component in nx.weakly_connected_components(dag):
        node_set = set(component)

        # 水平方向のDFS（R方向）
        h_root_headers = _find_root_headers(dag, "R", node_set)
        h_kv_items = []
        h_kv_cells = {}
        for root_id in h_root_headers:
            _dfs_collect_kv(dag, root_id, [], h_kv_items, cells, h_kv_cells, "R")

        # 垂直方向のDFS（D方向）
        v_root_headers = _find_root_headers(dag, "D", node_set)
        v_kv_items = []
        v_kv_cells = {}
        for root_id in v_root_headers:
            _dfs_collect_kv(dag, root_id, [], v_kv_items, cells, v_kv_cells, "D")

        # クラスタ内で葉ノード（value）が多い方を採用
        h_leaves = len({kv.value for kv in h_kv_items})
        v_leaves = len({kv.value for kv in v_kv_items})

        if v_leaves > h_leaves:
            kv_items.extend(v_kv_items)
            kv_cells.update(v_kv_cells)
            remove_dirs = ("R", "L")
        else:
            kv_items.extend(h_kv_items)
            kv_cells.update(h_kv_cells)
            remove_dirs = ("D", "U")

        # 採用されなかった方向のエッジをクラスタ内から削除
        edges_to_remove = [
            (u, v)
            for u, v, d in dag.edges(node_set, data=True)
            if v in node_set and d.get("dir") in remove_dirs
        ]
        dag.remove_edges_from(edges_to_remove)

    # DFSで到達できなかったcell/emptyはキーなしで追加
    visited_values = {kv.value for kv in kv_items}
    for cell in clustered_nodes["cell"] + clustered_nodes["empty"]:
        if cell.id not in visited_values:
            kv_items.append(KvItemSchema(id=None, key=[], value=cell.id, box=cell.box))
            kv_cells[cell.id] = cells[cell.id]

    return kv_items, dag, kv_cells
