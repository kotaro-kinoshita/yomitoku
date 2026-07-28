import networkx as nx

from .schemas.table_semantic_parser import KvItemSchema
from .utils.misc import (
    calc_overlap_ratio,
    is_bottom_adjacent,
    is_contained,
    is_right_adjacent,
)

def _assign_cells_to_regions(nodes, regions):
    """
    header / cell / empty ノードを、それを内包する領域(kv_item 領域)に割り当てる。

    - cell / empty (値セル) は最も重複率の高い1領域に属する（N:1）。
    - header (キー) は複数行にまたがる行ヘッダーになり得るため、内包される
      全領域に属する（N:M）。header_to_regions は header.id -> set(region_id)。

    どの領域にも属さないノードは割り当てず、グラフの構築対象から外す。
    """

    cell_to_region = {}
    header_to_regions = {}

    if len(regions) == 0:
        return cell_to_region, header_to_regions

    # セルと領域のマッチング（各セルを最も重複率の高い1領域に割り当て）
    for cell in nodes["cell"] + nodes["empty"]:
        best_region_id = None
        best_ratio = 0.0
        for region in regions:
            if is_contained(region.box, cell.box, threshold=0.2):
                ratio = calc_overlap_ratio(cell.box, region.box)[0]
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_region_id = region.id
        if best_region_id is not None:
            cell_to_region[cell.id] = best_region_id

    # ヘッダーと領域のマッチング（内包される全領域に属する）
    for header in nodes["header"]:
        regs = {
            region.id
            for region in regions
            if is_contained(region.box, header.box, threshold=0.2)
        }
        if regs:
            header_to_regions[header.id] = regs

    return cell_to_region, header_to_regions


def _attach_left_adjacent_orphan_headers(nodes, header_to_regions):
    """どのkv領域にも属さないヘッダーを、そのヘッダーが左方に隣接する
    「領域に属するヘッダー」の領域へ取り込む（先頭キー・行ヘッダーの取りこぼし救済）。

    例1: kv_item のboxが右側のキーしか覆っておらず、左端の上位キー(ヘッダー)が
    どの領域にも属していない場合に、右隣の領域ヘッダーと同じ領域へ結合する。

    例2: 複数行にまたがる左端の行ヘッダー(例:「控除対象扶養親族等」や行番号)は
    どの領域ともx方向で重ならず孤児になる。この場合、右隣に接する全ての領域
    ヘッダーの領域を「すべて」取り込むことで、各行(領域)の子と1:Nで接続できる
    ようにする。1領域のみへ結合すると1つの子にしか繋がらない不具合を防ぐ。

    orphan → orphan → 領域ヘッダー のような連鎖にも対応するため、
    変化が無くなるまで繰り返す。

    NOTE: 右隣接ヘッダーの領域は「単調に増える」形で毎パス再計算する。1回目に
    見つかった領域で固定してしまうと、右隣ヘッダーが後のパスで領域を獲得しても
    取りこぼす（例: 行ヘッダーが最初に領域を持っていた1つの行番号にしか繋がらない）。
    """
    headers = nodes["header"]

    # 直接どの領域にも属さなかったヘッダー(=孤児)のみを成長対象とする。
    # 直接割当済みヘッダーの領域は変更しない。
    orphan_ids = {h.id for h in headers if not header_to_regions.get(h.id)}

    updated = True
    while updated:
        updated = False
        for orphan in headers:
            if orphan.id not in orphan_ids:
                continue

            # orphan が左方に隣接する「領域に属するヘッダー」の領域を全て集める
            # (= 領域ヘッダーが orphan の右隣に隣接する)。毎パス全体から再計算し、
            # 右隣ヘッダーが後から獲得した領域も取り込めるようにする。
            attached = set(header_to_regions.get(orphan.id, set()))
            for other in headers:
                if other.id == orphan.id:
                    continue
                other_regions = header_to_regions.get(other.id)
                if not other_regions:
                    continue
                if is_right_adjacent(orphan.box, other.box):
                    attached |= other_regions

            if attached and attached != header_to_regions.get(orphan.id):
                header_to_regions[orphan.id] = attached
                updated = True

    return header_to_regions


def _calc_adjacent_header_to_cell(
    dag, cell_to_region, header_to_regions, headers, cells
):
    """
    同一領域内のヘッダーとセルの隣接関係を計算
    """

    for header in headers:
        header_regions = header_to_regions.get(header.id)
        if not header_regions:
            continue

        for cell in cells:
            cell_region_id = cell_to_region.get(cell.id, None)
            if cell_region_id is None:
                continue

            if cell_region_id not in header_regions:
                continue

            if is_right_adjacent(header.box, cell.box):
                dag.add_edge(header.id, cell.id, dir="R")
                dag.add_edge(cell.id, header.id, dir="L")

            if is_bottom_adjacent(header.box, cell.box):
                dag.add_edge(header.id, cell.id, dir="D")
                dag.add_edge(cell.id, header.id, dir="U")


def _calc_adjacent_header_to_header(dag, header_to_regions, nodes):
    """
    同一領域内のヘッダー同士の隣接関係を計算
    """

    for node in nodes:
        for potential_parent in nodes:
            if node.id == potential_parent.id:
                continue

            potential_parent_regions = header_to_regions.get(potential_parent.id)
            node_regions = header_to_regions.get(node.id)

            if not potential_parent_regions or not node_regions:
                continue

            # 領域を1つでも共有していれば同一グループとみなす
            if not (potential_parent_regions & node_regions):
                continue

            # 左右の隣接判定
            if is_right_adjacent(potential_parent.box, node.box):
                dag.add_edge(potential_parent.id, node.id, dir="R")
                dag.add_edge(node.id, potential_parent.id, dir="L")

            # 上下の隣接判定
            if is_bottom_adjacent(potential_parent.box, node.box):
                dag.add_edge(potential_parent.id, node.id, dir="D")
                dag.add_edge(node.id, potential_parent.id, dir="U")


def _calc_adjacent_cell_to_cell(dag, cell_to_region, nodes):
    """
    同一領域内のセル同士の隣接関係を計算
    """

    for node in nodes:
        node_region_id = cell_to_region.get(node.id, None)
        if node_region_id is None:
            continue

        for potential_parent in nodes:
            if node.id == potential_parent.id:
                continue

            potential_parent_region_id = cell_to_region.get(potential_parent.id, None)
            if potential_parent_region_id is None:
                continue

            if node_region_id != potential_parent_region_id:
                continue

            # 左右の隣接判定
            if is_right_adjacent(potential_parent.box, node.box):
                dag.add_edge(potential_parent.id, node.id, dir="R")
                dag.add_edge(node.id, potential_parent.id, dir="L")

            # 上下の隣接判定
            if is_bottom_adjacent(potential_parent.box, node.box):
                dag.add_edge(potential_parent.id, node.id, dir="D")
                dag.add_edge(node.id, potential_parent.id, dir="U")


def get_kv_items_dag(nodes, regions):
    cell_to_region, header_to_regions = _assign_cells_to_regions(nodes, regions)

    # どの領域にも属さないヘッダーを、左方に隣接する領域ヘッダーの領域へ取り込む
    header_to_regions = _attach_left_adjacent_orphan_headers(nodes, header_to_regions)

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
        cell_to_region,
        header_to_regions,
        nodes["header"],
        nodes["cell"],
    )

    _calc_adjacent_header_to_cell(
        dag,
        cell_to_region,
        header_to_regions,
        nodes["header"],
        nodes["empty"],
    )

    _calc_adjacent_header_to_header(dag, header_to_regions, nodes["header"])
    _calc_adjacent_cell_to_cell(dag, cell_to_region, nodes["cell"])

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


def parse_kv_items(nodes, cells, regions):
    dag = get_kv_items_dag(nodes, regions)

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
    for cell in nodes["cell"] + nodes["empty"]:
        if cell.id not in visited_values:
            kv_items.append(KvItemSchema(id=None, key=[], value=cell.id, box=cell.box))
            kv_cells[cell.id] = cells[cell.id]

    return kv_items, dag, kv_cells
