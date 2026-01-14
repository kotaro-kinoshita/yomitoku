import asyncio

import networkx as nx

from concurrent.futures import ThreadPoolExecutor

from .table_detector import TableDetector
from .cell_detector import CellDetector
from .text_detector import TextDetector
from .text_recognizer import TextRecognizer
from .document_analyzer import extract_words_within_element
from .ocr import OCRSchema, ocr_aggregate
from .utils.misc import is_contained, is_right_adjacent, is_bottom_adjacent

from .schemas.table_semantic_parser import TableGridSchema, KvItemSchema

from collections import deque

from .utils.visualizer import det_visualizer, dag_visualizer, cell_detector_visualizer
from .utils.misc import (
    replace_spanning_words_with_clipped_polys_poly,
    build_text_detector_schema_from_split_words_rotated_quad,
    box_to_poly,
    quad_to_poly,
    calc_overlap_ratio,
)

from .constants import PALETTE
from itertools import count

from .schemas.table_semantic_parser import (
    TableSemanticContentsSchema,
    TableSemanticParserSchema,
)

from .schemas.document_analyzer import TextDetectorSchema


from typing import List, Tuple


BBox = Tuple[float, float, float, float]


def _split_nodes_with_role(cells):
    """
    セルの役割ごとにノードを分割
    """

    nodes = {
        "header": [],
        "group": [],
        "cell": [],
        "empty": [],
    }
    for cell in cells:
        if cell.role not in nodes:
            nodes[cell.role] = []
        nodes[cell.role].append(cell)

    return nodes


def get_cell_by_id(cells, cell_id):
    for cell in cells:
        if cell.id == cell_id:
            return cell
    return None


def _components_with_isolates(dag, dir_value: str):
    """
    dir==dir_value のエッジだけで無向グラフを作り、連結成分を返す。
    孤立ノードも成分として含めるため、全ノードを先に add_node する。
    """
    G = nx.Graph()
    G.add_nodes_from(dag.nodes())
    G.add_edges_from(
        (u, v) for u, v, d in dag.edges(data=True) if d.get("dir") == dir_value
    )
    return list(nx.connected_components(G))


def cluster_heads_by_in_degree(dag: nx.DiGraph, dir_value: str):
    # 無向クラスタ
    comps = _components_with_isolates(dag, dir_value)

    # dir_value エッジだけ抽出した有向部分グラフ
    H = nx.DiGraph(
        (u, v, d) for u, v, d in dag.edges(data=True) if d.get("dir") == dir_value
    )
    H.add_nodes_from(dag.nodes())

    heads = []
    for comp in comps:
        # comp 内で in_degree==0 のノード（root）を全取得
        cand = [n for n in comp if H.in_degree(n) == 0]
        if cand:
            heads.extend(sorted(cand))
        else:
            # ループ等で入口がないケース
            heads.append(min(comp))

    return heads, comps


def split_bbox_by_right_neighbors_exact(
    G,
    u,
    right_nodes: List,
    bbox_key="bbox",
) -> List[BBox]:
    """
    u の bbox を、右隣セル(right_nodes)の bbox の y1/y2 を使って縦分割する。
    - right_nodes を上→下にソート
    - それぞれの (y1,y2) を u の範囲にクリップして採用
    - 採用区間が重なったり隙間が出たら、上から順に「連続な区間」に補正
    """
    ux1, uy1, ux2, uy2 = G.nodes[u][bbox_key]
    if not right_nodes:
        return [(ux1, uy1, ux2, uy2)]

    # 右隣の bbox を上→下に
    rights = sorted(
        right_nodes,
        key=lambda n: (G.nodes[n][bbox_key][1] + G.nodes[n][bbox_key][3]) / 2.0,
    )

    # 右隣の y区間を u 範囲にクリップして取得
    intervals = []
    for n in rights:
        _, ry1, _, ry2 = G.nodes[n][bbox_key]
        a = max(uy1, ry1)
        b = min(uy2, ry2)
        intervals.append((a, b))

    # 連続区間に補正（ズレ対策）
    # 1) 無効区間（a>=b）は捨てずに後で埋めるためプレースホルダとして残す
    fixed = []
    cur = uy1
    for a, b in intervals:
        a = max(a, cur)  # 前の終端より上は切り上げ
        b = max(b, a)  # b>=a に
        fixed.append([a, b])
        cur = b

    # 2) 末尾が uy2 に届かないなら、最後を伸ばす
    if fixed:
        fixed[-1][1] = uy2

    # 3) 途中で長さ0があるなら、隣の区間から分配（簡易）
    #    （本当に0が出るのは right bbox が u とほぼ重ならない時＝入力が壊れてる時）
    for i in range(len(fixed)):
        a, b = fixed[i]
        if b - a <= 1e-3:
            # 可能なら次区間の先頭を少し削って埋める
            if i + 1 < len(fixed) and fixed[i + 1][1] - fixed[i + 1][0] > 2e-3:
                take = (fixed[i + 1][1] - fixed[i + 1][0]) * 0.1
                fixed[i][1] = fixed[i][0] + take
                fixed[i + 1][0] = fixed[i][1]
            # それも無理なら等分で埋める（最終手段）
            else:
                pass

    # bboxへ
    return [(ux1, a, ux2, b) for a, b in fixed]


def split_bbox_by_down_neighbors_exact_x(
    G,
    u,
    down_nodes: List,
    bbox_key="bbox",
) -> List[BBox]:
    """
    u の bbox を、下隣セル(down_nodes)の bbox の x1/x2 を使って横分割する（x方向）。
    - down_nodes を左→右にソート（x中心）
    - それぞれの (x1,x2) を u の範囲にクリップして採用
    - 採用区間が重なったり隙間が出たら、左から順に「連続な区間」に補正
    """
    ux1, uy1, ux2, uy2 = G.nodes[u][bbox_key]
    if not down_nodes:
        return [(ux1, uy1, ux2, uy2)]

    # 下隣の bbox を左→右に
    downs = sorted(
        down_nodes,
        key=lambda n: (G.nodes[n][bbox_key][0] + G.nodes[n][bbox_key][2]) / 2.0,
    )

    # 下隣の x区間を u 範囲にクリップして取得
    intervals = []
    for n in downs:
        dx1, _, dx2, _ = G.nodes[n][bbox_key]
        a = max(ux1, dx1)
        b = min(ux2, dx2)
        intervals.append((a, b))

    # 連続区間に補正（ズレ対策）
    fixed = []
    cur = ux1
    for a, b in intervals:
        a = max(a, cur)  # 前の終端より左は切り上げ
        b = max(b, a)  # b>=a に
        fixed.append([a, b])
        cur = b

    # 末尾が ux2 に届かないなら、最後を伸ばす
    if fixed:
        fixed[-1][1] = ux2

    # 途中で長さ0があるなら、隣の区間から分配（簡易）
    for i in range(len(fixed)):
        a, b = fixed[i]
        if b - a <= 1e-3:
            if i + 1 < len(fixed) and fixed[i + 1][1] - fixed[i + 1][0] > 2e-3:
                take = (fixed[i + 1][1] - fixed[i + 1][0]) * 0.1
                fixed[i][1] = fixed[i][0] + take
                fixed[i + 1][0] = fixed[i][1]
            else:
                pass

    # bboxへ（xだけ分割、yはそのまま）
    return [(a, uy1, b, uy2) for a, b in fixed]


def normalize_row_with_out_edges(
    dag: nx.DiGraph,
    head: str,
    dir_key: str = "dir",
    out_edge_type: str = "R",
    in_edge_type: str = "L",
) -> nx.DiGraph:
    """
    head から辿れるノードを対象に、横方向(out_edge_type/in_edge_type)で
    out が複数あるノードを分割して 1:1 化する。

    - out_edge_type="R" のとき: 右方向に分割（従来）
    - out_edge_type="L" のとき: 左方向に分割（左右反転対応）
    """
    G = dag.copy()
    queue = deque([head])
    dup_counter = count(1)

    while queue:
        u = queue.popleft()
        if u not in G.nodes:
            continue

        # 前方(out) と後方(bwd) を、向きに応じて正しく取る
        outs_fwd = [v for v in G.successors(u) if G[u][v].get(dir_key) == out_edge_type]

        if out_edge_type == "R":
            # 左側セル（左→u が R）
            outs_bwd = [p for p in G.predecessors(u) if G[p][u].get(dir_key) == "R"]

            def ok_adj(bwd, dup):
                # bwd(左) は dup の左にいるはず
                return is_right_adjacent(
                    G.nodes[bwd]["bbox"], G.nodes[dup]["bbox"], rule="soft"
                )

        elif out_edge_type == "L":
            # 右側セル（右→u が L）
            outs_bwd = [p for p in G.predecessors(u) if G[p][u].get(dir_key) == "L"]

            def ok_adj(bwd, dup):
                # bwd(右) は dup の右にいるはず（判定向きを反転）
                return is_right_adjacent(
                    G.nodes[dup]["bbox"], G.nodes[bwd]["bbox"], rule="soft"
                )

        else:
            outs_bwd = []

            def ok_adj(bwd, dup):
                return False

        # 縦方向（D/U）は常に同じ定義：上→下がD、下→上がU
        up_cells = [p for p in G.predecessors(u) if G[p][u].get(dir_key) == "D"]  # 上→u
        down_cells = [v for v in G.successors(u) if G[u][v].get(dir_key) == "D"]  # u→下

        if len(outs_fwd) > 1:
            seg_bboxes = split_bbox_by_right_neighbors_exact(
                G,
                u,
                outs_fwd,
                bbox_key="bbox",
            )

            seg_bboxes = sorted(seg_bboxes, key=lambda box: box[1])
            outs_fwd = sorted(outs_fwd, key=lambda n: G.nodes[n]["bbox"][1])

            base_attr = dict(G.nodes[u])
            dups = []

            # 各分割 bbox ごとに新ノードを作成（衝突しない名前）
            for bb in seg_bboxes:
                nu = f"{u}__dup{next(dup_counter)}"
                attr = dict(base_attr)
                attr["bbox"] = bb
                G.add_node(nu, **attr)
                dups.append(nu)

            # 前方(out) と複製ノードを 1:1 で接続
            for out, dup in zip(outs_fwd, dups):
                G.add_edge(dup, out, dir=out_edge_type)
                G.add_edge(out, dup, dir=in_edge_type)

            # 先頭ノードは上方セルと接続（U/D）
            for p in up_cells:
                G.add_edge(dups[0], p, dir="U")
                G.add_edge(p, dups[0], dir="D")

            # 末尾ノードは下方セルと接続（D/U）
            for p in down_cells:
                G.add_edge(dups[-1], p, dir="D")
                G.add_edge(p, dups[-1], dir="U")

            # 複製ノード同士を縦に接続（D/U）
            for a, b in zip(dups, dups[1:]):
                G.add_edge(a, b, dir="D")
                G.add_edge(b, a, dir="U")

            # 後方（反対側）隣接ノードとも接続（向きに応じて判定を切替）
            for bwd in outs_bwd:
                for dup in dups:
                    if ok_adj(bwd, dup):
                        G.add_edge(bwd, dup, dir=out_edge_type)
                        G.add_edge(dup, bwd, dir=in_edge_type)
                        queue.append(bwd)

            # 元ノード削除
            G.remove_node(u)

            # 生成dupを再探索
            for dup in dups:
                queue.append(dup)

        else:
            # 次へ
            for v in outs_fwd:
                queue.append(v)

    return G


def normalize_col_with_out_edges(
    dag: nx.DiGraph,
    head: str,
    dir_key: str = "dir",
    out_edge_type: str = "D",
    in_edge_type: str = "U",
) -> nx.DiGraph:
    """
    head から辿れるノードを対象に、右隣セルが複数あるノードを分割して 1:1 化する。
    """
    G = dag.copy()
    queue = deque([head])
    dup_counter = count(1)

    while queue:
        u = queue.popleft()
        if u not in G.nodes:
            continue

        outs_fwd = [v for v in G.successors(u) if G[u][v].get(dir_key) == out_edge_type]
        if out_edge_type == "D":
            outs_bwd = [p for p in G.predecessors(u) if G[p][u].get(dir_key) == "D"]

            def ok_adj(bwd, dup):
                return is_bottom_adjacent(
                    G.nodes[bwd]["bbox"], G.nodes[dup]["bbox"], rule="soft"
                )
        elif out_edge_type == "U":
            outs_bwd = [v for v in G.successors(u) if G[u][v].get(dir_key) == "D"]

            def ok_adj(bwd, dup):
                return is_bottom_adjacent(
                    G.nodes[dup]["bbox"], G.nodes[bwd]["bbox"], rule="soft"
                )
        else:
            outs_bwd = []

            def ok_adj(bwd, dup):
                return False

        left_cells = [p for p in G.predecessors(u) if G[p][u].get(dir_key) == "R"]
        right_cells = [v for v in G.successors(u) if G[u][v].get(dir_key) == "R"]
        if len(outs_fwd) > 1:
            seg_bboxes = split_bbox_by_down_neighbors_exact_x(
                G,
                u,
                outs_fwd,
                bbox_key="bbox",
            )

            seg_bboxes = sorted(seg_bboxes, key=lambda box: box[0])
            outs_fwd = sorted(outs_fwd, key=lambda n: G.nodes[n]["bbox"][0])

            base_attr = dict(G.nodes[u])
            dups = []

            # 各分割 bbox ごとに新ノードを作成
            for i, bb in enumerate(seg_bboxes):
                nu = f"{u}__dup{next(dup_counter)}"
                attr = dict(base_attr)
                attr["bbox"] = bb
                G.add_node(nu, **attr)
                dups.append(nu)

            # 隣接ノードと複製ノードを接続
            for i, (out, dup) in enumerate(zip(outs_fwd, dups)):
                G.add_edge(dup, out, dir=out_edge_type)
                G.add_edge(out, dup, dir=in_edge_type)

            # 先頭ノードは左セルと接続
            for p in left_cells:
                G.add_edge(dups[0], p, dir="L")
                G.add_edge(p, dups[0], dir="R")

            # 末尾ノードは右セルと接続
            for p in right_cells:
                G.add_edge(dups[-1], p, dir="R")
                G.add_edge(p, dups[-1], dir="L")

            # 複製ノード同士を接続
            for i in range(len(dups) - 1):
                G.add_edge(dups[i], dups[i + 1], dir="R")
                G.add_edge(dups[i + 1], dups[i], dir="L")

            # 後方隣接ノードとも接続
            for bwd in outs_bwd:
                for dup in dups:
                    if ok_adj(bwd, dup):
                        G.add_edge(bwd, dup, dir=out_edge_type)
                        G.add_edge(dup, bwd, dir=in_edge_type)

                        # 後方ノードは再探索
                        queue.append(bwd)

            # 元ノード削除
            G.remove_node(u)
            for dup in dups:
                queue.append(dup)
        else:
            for v in outs_fwd:
                queue.append(v)

    return G


def expand_dir_to_uit_row(
    dag: nx.DiGraph,
    dir_key: str = "dir",
) -> nx.DiGraph:
    """
    target_dir（クラスタ抽出に使う方向）で線クラスタの head を取り、
    その head から edge_fwd/edge_bwd のペアを 1:1 化する。
    """
    G = dag.copy()

    line_heads, line_clusters = cluster_heads_by_in_degree(G, dir_value="R")
    for head in line_heads:
        G = normalize_row_with_out_edges(
            G,
            head,
            dir_key=dir_key,
            in_edge_type="L",
            out_edge_type="R",
        )

    line_heads, line_clusters = cluster_heads_by_in_degree(G, dir_value="L")
    for head in line_heads:
        G = normalize_row_with_out_edges(
            G,
            head,
            dir_key=dir_key,
            in_edge_type="R",
            out_edge_type="L",
        )

    return G


def expand_dir_to_uit_col(
    dag: nx.DiGraph,
    dir_key: str = "dir",
) -> nx.DiGraph:
    """
    target_dir（クラスタ抽出に使う方向）で線クラスタの head を取り、
    その head から edge_fwd/edge_bwd のペアを 1:1 化する。
    """
    G = dag.copy()

    line_heads, line_clusters = cluster_heads_by_in_degree(G, dir_value="D")
    for head in line_heads:
        G = normalize_col_with_out_edges(
            G,
            head,
            dir_key=dir_key,
            in_edge_type="U",
            out_edge_type="D",
        )

    line_heads, line_clusters = cluster_heads_by_in_degree(G, dir_value="U")
    for head in line_heads:
        G = normalize_col_with_out_edges(
            G,
            head,
            dir_key=dir_key,
            in_edge_type="D",
            out_edge_type="U",
        )

    return G


def cluster_nodes(clusters, nodes):
    clustered_nodes_list = []

    for i, cluster in enumerate(clusters):
        clustered_nodes = {
            "header": [],
            "cell": [],
            "empty": [],
        }

        for id in cluster:
            node = get_cell_by_id(nodes["header"] + nodes["cell"] + nodes["empty"], id)
            clustered_nodes[node.role].append(node)

        clustered_nodes_list.append(clustered_nodes)

    return clustered_nodes_list


def drop_single_out_edge_by_type(G: nx.DiGraph, edge_type: str, type_key: str = "type"):
    """
    各ノード u について:
      u から出るエッジのうち type == edge_type の本数が 1 本なら
      その 1 本を削除する
    """
    to_remove = []
    for u in G.nodes():
        outs = [(u, v) for v in G.successors(u) if G[u][v].get(type_key) == edge_type]
        if len(outs) == 1:
            to_remove.append(outs[0])

    G.remove_edges_from(to_remove)
    return to_remove


def replace_edge_type(G, from_type, to_type, key="type"):
    for _, _, d in G.edges(data=True):
        if d.get(key) == from_type:
            d[key] = to_type


def weakly_cluster_nodes_with_graph(nodes):
    """
    ビューリスティックにセル間の関係をDAGで表現し、弱連結成分でクラスタリング
    """

    dag = nx.DiGraph()

    for role in nodes:
        for node in nodes[role]:
            if role not in ["header", "cell", "empty"]:
                continue

            dag.add_node(node.id, bbox=node.box, role=node.role)

    for header in nodes["header"]:
        for cell in nodes["cell"] + nodes["empty"]:
            if is_bottom_adjacent(
                header.box,
                cell.box,
                rule="nest",
            ):
                dag.add_edge(header.id, cell.id, dir="D")

            if is_right_adjacent(
                header.box,
                cell.box,
                rule="soft",
            ):
                dag.add_edge(header.id, cell.id, dir="D")

        for header2 in nodes["header"]:
            if header.id == header2.id:
                continue

            if is_right_adjacent(
                header.box,
                header2.box,
                rule="soft",
            ):
                dag.add_edge(header.id, header2.id, dir="D")

            if is_bottom_adjacent(
                header.box,
                header2.box,
                rule="child",
            ):
                dag.add_edge(header.id, header2.id, dir="nest")

    # ヘッダーの縦の1:1結合はヒューリスティック的にまれにしか起きないので削除
    drop_single_out_edge_by_type(dag, edge_type="nest", type_key="dir")
    replace_edge_type(dag, from_type="nest", to_type="D", key="dir")

    for cell1 in nodes["cell"] + nodes["empty"]:
        for cell2 in nodes["cell"] + nodes["empty"]:
            if cell1.id == cell2.id:
                continue

            if is_right_adjacent(
                cell1.box,
                cell2.box,
                rule="soft",
            ):
                dag.add_edge(cell1.id, cell2.id, dir="D")

            if is_bottom_adjacent(
                cell1.box,
                cell2.box,
                rule="hard",
            ):
                dag.add_edge(cell1.id, cell2.id, dir="D")

    for empty in nodes["empty"]:
        for header in nodes["header"]:
            if is_bottom_adjacent(
                empty.box,
                header.box,
                rule="hard",
            ):
                dag.add_edge(empty.id, header.id, dir="D")
            if is_right_adjacent(
                empty.box,
                header.box,
                rule="hard",
            ):
                dag.add_edge(empty.id, header.id, dir="D")

    return list(nx.weakly_connected_components(dag))


def is_grid_cluster(nodes):
    G = nx.DiGraph()
    for cell in nodes["cell"] + nodes["empty"]:
        G.add_node(cell.id, bbox=cell.box, role=cell.role)

    for cell1 in nodes["cell"] + nodes["empty"]:
        for cell2 in nodes["cell"] + nodes["empty"]:
            if cell1.id == cell2.id:
                continue

            if is_bottom_adjacent(
                cell1.box,
                cell2.box,
                rule="hard",
            ):
                G.add_edge(cell1.id, cell2.id, dir="V")

            if is_right_adjacent(
                cell1.box,
                cell2.box,
                rule="hard",
            ):
                G.add_edge(cell1.id, cell2.id, dir="H")

    VG = nx.Graph((u, v, d) for u, v, d in G.edges(data=True) if d.get("dir") == "V")
    HG = nx.Graph((u, v, d) for u, v, d in G.edges(data=True) if d.get("dir") == "H")

    h_components = list(nx.connected_components(HG))
    v_components = list(nx.connected_components(VG))

    # 2列以上かつ2行以上で構成されるものをグリッドと判断
    if len(h_components) > 1 and len(v_components) > 1:
        return True

    return False


def get_grid_dag(nodes):
    dag = nx.DiGraph()
    cells = nodes["cell"] + nodes["empty"] + nodes["header"]

    for cell in cells:
        dag.add_node(
            cell.id,
            bbox=cell.box,
            role=cell.role,
            contents=cell.contents,
        )

    for cell1 in cells:
        for cell2 in cells:
            if cell1.id == cell2.id:
                continue

            if is_bottom_adjacent(
                cell1.box,
                cell2.box,
                rule="soft",
            ):
                dag.add_edge(cell1.id, cell2.id, dir="D")
                dag.add_edge(cell2.id, cell1.id, dir="U")

            if is_right_adjacent(
                cell1.box,
                cell2.box,
                rule="soft",
            ):
                dag.add_edge(cell1.id, cell2.id, dir="R")
                dag.add_edge(cell2.id, cell1.id, dir="L")

    return dag


def expand_grid_to_unit(dag: nx.DiGraph) -> nx.DiGraph:
    dag = expand_dir_to_uit_row(dag)
    dag = expand_dir_to_uit_col(dag)
    return dag


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


def _calc_spans_and_indices_from_raw_grid(raw_data):
    """
    raw_data: NxM の node_id 行列（欠損は None）
    return:
      info: {
        cell_id: {
          "row": int, "col": int,          # 0-start 左上（アンカー）
          "row_span": int, "col_span": int
        }
      }
    """
    pos = {}
    for r, row in enumerate(raw_data):
        for c, cell_id in enumerate(row):
            if cell_id is None:
                continue
            if cell_id not in pos:
                pos[cell_id] = [r, r, c, c]  # rmin,rmax,cmin,cmax
            else:
                pos[cell_id][0] = min(pos[cell_id][0], r)
                pos[cell_id][1] = max(pos[cell_id][1], r)
                pos[cell_id][2] = min(pos[cell_id][2], c)
                pos[cell_id][3] = max(pos[cell_id][3], c)

    info = {}
    for cell_id, (rmin, rmax, cmin, cmax) in pos.items():
        info[cell_id] = {
            "row": rmin,  # 0-start
            "col": cmin,  # 0-start
            "row_span": (rmax - rmin + 1),
            "col_span": (cmax - cmin + 1),
        }

    return info


class TableSemanticParser:
    def __init__(
        self, configs={}, device="cuda:1", visualize=False, dag_visualize=True
    ):
        table_detector_kwargs = {
            "device": device,
            "visualize": visualize,
        }
        table_parser_kwargs = {
            "device": device,
            "visualize": visualize,
        }

        text_detector_kwargs = {
            "device": device,
            "visualize": visualize,
        }

        text_recognizer_kwargs = {
            "device": device,
            "visualize": visualize,
        }

        if isinstance(configs, dict):
            if "table_detector" in configs:
                table_detector_kwargs.update(configs["table_detector"])

            if "table_parser" in configs:
                table_parser_kwargs.update(configs["table_parser"])

            if "text_detector" in configs:
                text_detector_kwargs.update(configs["text_detector"])

            if "text_recognizer" in configs:
                text_recognizer_kwargs.update(configs["text_recognizer"])
        else:
            raise ValueError(
                "configs must be a dict. See the https://kotaro-kinoshita.github.io/yomitoku-dev/usage/"
            )

        self.layout_parser = TableDetector(
            **table_detector_kwargs,
        )
        self.cell_detector = CellDetector(
            **table_parser_kwargs,
        )

        self.text_detector = TextDetector(
            **text_detector_kwargs,
        )

        self.text_recognizer = TextRecognizer(
            **text_recognizer_kwargs,
        )

        self.visualize = visualize
        self.dag_visualize = dag_visualize

    def aggregate(self, ocr_res, cells):
        """
        セル領域内のOCR結果を集約してセルの内容として設定
        """

        for cell in cells:
            words, direction, _ = extract_words_within_element(ocr_res.words, cell)

            if words is None:
                words = ""

            words = words.replace("\n", "").strip()
            cell.contents = words

    async def run_models(self, img):
        with ThreadPoolExecutor(max_workers=2) as executor:
            loop = asyncio.get_running_loop()
            tasks = [
                loop.run_in_executor(executor, self.text_detector, img),
                loop.run_in_executor(executor, self.layout_parser, img),
            ]

            results = await asyncio.gather(*tasks)

        results_det, _ = results[0]
        tables, vis_layout = results[1]

        results_table, vis_cell, vis_group = self.cell_detector(
            img, tables, vis=vis_layout
        )

        word_dicts = [
            {"poly": quad_to_poly(q), "score": s}
            for q, s in zip(results_det.points, results_det.scores)
        ]

        cell_dicts = [
            {"id": str(t.id), "poly": box_to_poly(t.box)} for t in results_table
        ]

        # セルにまたがるテキスト領域の分割
        split_words = replace_spanning_words_with_clipped_polys_poly(
            words=word_dicts,
            cells=cell_dicts,
            min_area_ratio=0.03,
            keep_unsplit=True,
        )

        schema_dict = build_text_detector_schema_from_split_words_rotated_quad(
            split_words
        )
        results_det = TextDetectorSchema(**schema_dict)

        vis_det = None
        if self.visualize:
            vis_det = det_visualizer(
                img,
                results_det.points,
            )

        results_rec, vis_ocr = self.text_recognizer(
            img, results_det.points, vis=vis_det
        )
        outputs = {"words": ocr_aggregate(results_det, results_rec)}
        results_ocr = OCRSchema(**outputs)

        return results_ocr, results_table, vis_cell, vis_layout, vis_ocr

    def get_line_with_head(
        self, dag: nx.DiGraph, head: str, dir_value: str
    ) -> List[str]:
        """
        head から辿れるノードを dir_value エッジで取得
        """
        line_nodes = []
        queue = deque([head])

        while queue:
            u = queue.popleft()
            if u not in dag.nodes:
                continue

            line_nodes.append(u)

            for v in dag.successors(u):
                if dag[u][v].get("dir") == dir_value:
                    queue.append(v)

        return line_nodes

    def get_kv_items_dag(self, nodes, groups):
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

    def __call__(self, img, template=None):
        results_ocr, results_table, vis_cell, vis_layout, vis_ocr = asyncio.run(
            self.run_models(img)
        )

        semantic_info = []
        for table in results_table:
            self.aggregate(results_ocr, table.cells)

        for i, table in enumerate(results_table):
            table_information = {
                "id": table.id,
                "box": table.box,
                "cells": {cell.id: cell for cell in table.cells},
                "style": "border",
                "kv_items": [],
                "grids": [],
            }
            if template is None:
                nodes = _split_nodes_with_role(table.cells)
                clusters = weakly_cluster_nodes_with_graph(nodes)
                cluster_nodes_list = cluster_nodes(clusters, nodes)

                for cluster, clustered_nodes in zip(clusters, cluster_nodes_list):
                    if is_grid_cluster(clustered_nodes):
                        dag = get_grid_dag(clustered_nodes)
                        dag = expand_grid_to_unit(dag)
                        row_root = cluster_heads_by_in_degree(dag, dir_value="R")[0]
                        col_root = cluster_heads_by_in_degree(dag, dir_value="D")[0]

                        row_root = sorted(
                            row_root,
                            key=lambda n: dag.nodes[n]["bbox"][1],
                        )

                        col_root = sorted(
                            col_root,
                            key=lambda n: dag.nodes[n]["bbox"][0],
                        )

                        raw_data = []
                        for r_head in row_root:
                            row_nodes = self.get_line_with_head(
                                dag, r_head, dir_value="R"
                            )
                            row_nodes = sorted(
                                row_nodes,
                                key=lambda n: dag.nodes[n]["bbox"][0],
                            )

                            row = []
                            for c_head in col_root:
                                c_nodes = self.get_line_with_head(
                                    dag, c_head, dir_value="D"
                                )
                                c_nodes = sorted(
                                    c_nodes,
                                    key=lambda n: dag.nodes[n]["bbox"][1],
                                )

                                cell_id = set(row_nodes).intersection(set(c_nodes))
                                if cell_id:
                                    cell = list(cell_id)[0]
                                    row.append(list(cell_id)[0])
                                else:
                                    row.append(None)
                            raw_data.append(row)

                        roles = [
                            [
                                dag.nodes[cell_id]["role"]
                                if cell_id is not None
                                else "empty"
                                for cell_id in row
                            ]
                            for row in raw_data
                        ]

                        is_header_row = [
                            all(role == "header" or role == "empty" for role in row)
                            for row in roles
                        ]

                        data = []
                        for i, row in enumerate(raw_data):
                            data.append(
                                [
                                    cell_id.split("__dup")[0]
                                    if cell_id is not None
                                    else None
                                    for cell_id in row
                                ]
                            )

                        cell_info = _calc_spans_and_indices_from_raw_grid(data)
                        for cell_id, info in cell_info.items():
                            cell = table_information["cells"][cell_id]
                            cell.row = info["row"]
                            cell.col = info["col"]
                            cell.row_span = info["row_span"]
                            cell.col_span = info["col_span"]

                        n_cols = len(data[0])
                        n_rows = len(data)

                        col_headers = []
                        for col_idx in range(n_cols):
                            col_header = []
                            for row_idx in range(n_rows):
                                if is_header_row[row_idx]:
                                    if data[row_idx][col_idx] is not None:
                                        col_header.append(data[row_idx][col_idx])
                            col_headers.append(list(set(col_header)))

                        data = [
                            raw for i, raw in enumerate(data) if not is_header_row[i]
                        ]

                        grid = TableGridSchema(
                            id=f"g{str(len(table_information['grids']))}",
                            n_row=len(data),
                            n_col=n_cols,
                            data=data,
                            col_headers=col_headers,
                        )

                        table_information["grids"].append(grid)

                        vis_cell = dag_visualizer(dag, vis_cell)

                    else:
                        dag = self.get_kv_items_dag(clustered_nodes, nodes["group"])
                        for cell in clustered_nodes["cell"]:
                            kv_items_row = self.get_line_with_head(
                                dag, cell.id, dir_value="L"
                            )

                            kv_items_col = self.get_line_with_head(
                                dag, cell.id, dir_value="U"
                            )

                            headers = []
                            headers.extend(
                                [
                                    dag.nodes[h]
                                    for h in kv_items_row
                                    if dag.nodes[h]["role"] == "header"
                                ]
                            )
                            headers.extend(
                                [
                                    dag.nodes[h]
                                    for h in kv_items_col
                                    if dag.nodes[h]["role"] == "header"
                                ]
                            )

                            kv_items = KvItemSchema(
                                key=[header["id"] for header in headers],
                                value=cell.id,
                            )

                            table_information["kv_items"].append(kv_items)

                            vis_cell = dag_visualizer(dag, vis_cell)

            table_information["kv_items"] = sorted(
                table_information["kv_items"],
                key=lambda kv: table_information["cells"][kv.value].box[1],
            )

            semantic_info.append(TableSemanticContentsSchema(**table_information))

        semantic_info = TableSemanticParserSchema(
            tables=semantic_info,
            words=results_ocr.words,
        )

        if template is not None:
            semantic_info.load_template_json(template)
            vis_cell = vis_layout.copy()
            for table in semantic_info.tables:
                vis_cell, _ = cell_detector_visualizer(
                    vis_cell,
                    vis_layout,
                    table.cells.values(),
                )

        return semantic_info, vis_cell, vis_ocr


def debug_grid_regions(img, regions):
    import cv2

    vis = img.copy()
    for region in regions:
        box = region["box"]
        cv2.rectangle(
            vis,
            (box[0], box[1]),
            (box[2], box[3]),
            (255, 0, 0),
            3,
        )
    return vis


def debug_cluster_nodes_with_graph(img, nodes, clusters):
    import cv2

    cells = nodes["header"] + nodes["cell"] + nodes["empty"]

    for i, cluster in enumerate(clusters):
        for id in cluster:
            print(id)
            node = get_cell_by_id(cells, id)
            box = node.box
            color = PALETTE[i % len(PALETTE)]
            cv2.rectangle(
                img,
                (box[0], box[1]),
                (box[2], box[3]),
                color,
                3,
            )

    return img
