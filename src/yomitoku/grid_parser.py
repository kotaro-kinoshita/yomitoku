import networkx as nx

from collections import deque
from itertools import count

from .schemas.table_semantic_parser import TableGridSchema
from .utils.misc import is_right_adjacent, is_bottom_adjacent, get_line_with_head

from typing import List, Tuple

from .utils.union_find import UnionFind


BBox = Tuple[float, float, float, float]

# 隣接判定しきい値のセル寸法に対する比率。
# 固定ピクセル値は画像解像度・表の行ピッチに依存して破綻する
# （例: 行高~20pxの表で dist_threshold=20 だと1行飛ばした先のセルまで
# 「下隣」と誤判定され、単位セル展開で幅0のスライバーが生成される）ため、
# クラスタ内セルの高さ・幅の下位25%点を基準にした相対値を用いる。
# - DIST_SCALE: 距離しきい値 = 寸法の下位25%点 × この比率。
#   1行(1列)分の寸法より十分小さいギャップのみを隣接とみなす。
# - IGNORE_SCALE: 対角(斜め)接触の棄却しきい値の比率。
# - DIST_FLOOR: 罫線太さ・検出ノイズを許容する下限(px)。
GRID_DIST_SCALE = 0.5
GRID_IGNORE_SCALE = 0.25
GRID_DIST_FLOOR = 3.0


def _lower_quantile(values, q=0.25):
    """下位q分位点（簡易・補間なし）を返す。"""
    vals = sorted(values)
    idx = min(int(len(vals) * q), len(vals) - 1)
    return vals[idx]


def _calc_adjacency_thresholds(boxes):
    """セル寸法の統計量から隣接判定しきい値を算出する。

    基準スケールは「セル高・セル幅それぞれの下位25%点の min」（=最小セル寸法の
    頑健推定）。外れ値（結合セル等の大きなセル）の影響を避けるため下位25%点を
    採用する。

    - dist: 基準スケールより十分小さいギャップのみ隣接とみなす。
      1行(1列)分の寸法より小さくすることで行・列を飛び越えた誤隣接を防ぐ。
    - ignore: 対角(斜め)接触の棄却しきい値。正対する隣接セルの角距離は
      最小セル寸法程度になるため、それより十分小さくする必要がある
      （寸法の大きい軸で計算すると正規の隣接まで棄却される）。
    """
    if not boxes:
        return {
            "dist": GRID_DIST_FLOOR,
            "ignore": GRID_DIST_FLOOR / 2,
        }

    h_scale = _lower_quantile([b[3] - b[1] for b in boxes])
    w_scale = _lower_quantile([b[2] - b[0] for b in boxes])
    scale = min(h_scale, w_scale)

    return {
        "dist": max(GRID_DIST_FLOOR, GRID_DIST_SCALE * scale),
        "ignore": max(GRID_DIST_FLOOR / 2, GRID_IGNORE_SCALE * scale),
    }


def _get_grid_dag(nodes, thresholds=None):
    dag = nx.DiGraph()
    cells = nodes["cell"] + nodes["empty"] + nodes["header"]

    if thresholds is None:
        thresholds = _calc_adjacency_thresholds([c.box for c in cells])

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
                dist_threshold=thresholds["dist"],
                ignore_dist_threshold=thresholds["ignore"],
                overlap_ratio_th=0.25,
            ):
                dag.add_edge(cell1.id, cell2.id, dir="D")
                dag.add_edge(cell2.id, cell1.id, dir="U")

            if is_right_adjacent(
                cell1.box,
                cell2.box,
                rule="soft",
                dist_threshold=thresholds["dist"],
                ignore_dist_threshold=thresholds["ignore"],
                overlap_ratio_th=0.25,
            ):
                dag.add_edge(cell1.id, cell2.id, dir="R")
                dag.add_edge(cell2.id, cell1.id, dir="L")

    return dag


# out 隣接セル同士の分割軸区間 IoU がこの値以上なら「同一スロットの重複検出」
# とみなしてグループ化する。片方がもう片方を包含するだけの正規のスパン関係
# （例: 帯ヘッダとその下の列ヘッダで IoU〜0.3）はグループ化せず、
# ほぼ同一区間（IoU〜1.0）の二重検出のみを対象にするための閾値。
OUT_GROUP_IOU_TH = 0.7


def _group_outs_by_axis_interval(G, outs, axis, iou_th=OUT_GROUP_IOU_TH):
    """out 隣接セルを分割軸の区間 IoU でグループ化し、区間中心順に返す。

    セル検出モデルがほぼ同じ領域に重複してセルを出すことがあり、その2つを
    別スロットとして 1:1 分割すると長さ0のスライバー複製ノードが生じ、
    エッジが不正な位置に潰れる。区間がほぼ一致する out は同一スロットに
    まとめ、1つの複製ノードへ N:1 で接続する。

    axis: 0 で x 区間（列分割）、1 で y 区間（行分割）。
    return: (groups, intervals)
      groups: ノードidのリストのリスト（区間中心の昇順）
      intervals: 各グループのメンバー区間の和集合 (a, b)
    """
    ivs = [
        (G.nodes[n]["bbox"][axis], G.nodes[n]["bbox"][axis + 2]) for n in outs
    ]

    parent = list(range(len(outs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(outs)):
        for j in range(i + 1, len(outs)):
            a1, a2 = ivs[i]
            b1, b2 = ivs[j]
            inter = max(0.0, min(a2, b2) - max(a1, b1))
            union = max(a2, b2) - min(a1, b1)
            if union > 0 and inter / union >= iou_th:
                parent[find(i)] = find(j)

    members = {}
    for i in range(len(outs)):
        members.setdefault(find(i), []).append(i)

    groups = []
    intervals = []
    for idxs in members.values():
        groups.append([outs[i] for i in idxs])
        intervals.append(
            (min(ivs[i][0] for i in idxs), max(ivs[i][1] for i in idxs))
        )

    order = sorted(
        range(len(groups)), key=lambda k: (intervals[k][0] + intervals[k][1]) / 2.0
    )
    return [groups[k] for k in order], [intervals[k] for k in order]


def _assign_slots_by_intervals(lo, hi, groups, intervals, min_len=None):
    """[lo, hi] を、中心順に並んだグループ区間に沿って連続なスロット列に
    割り当てる。

    - 各 interval を [lo, hi] にクリップし、先頭から順に「連続な区間」に補正
    - クリップ・補正後に取り分が min_len 以下しか残らないグループ
      （先行グループの区間に包含・消費されたもの。例: 融合誤検出セルが
      実セルを包含しているケースや、隣の値セルが列をまたいで検出された
      ケース）は、独立スロットにせず直前のスロットへ併合する。
      以前はここで長さ0〜数pxのスライバー区間が生成され、実体と無関係な
      位置から斜めエッジが張られたり、幻の行・列が生じていた。

    return: (segs, merged_groups)  両者は同じ長さ・同じ順序
    """
    if min_len is None:
        min_len = GRID_DIST_FLOOR

    segs = []
    merged_groups = []
    pending = []  # 先頭でスロットを確保できなかったグループの退避先
    cur = lo
    for group, (a, b) in zip(groups, intervals):
        a = max(lo, min(a, hi))
        b = max(lo, min(b, hi))
        a = max(a, cur)  # 前の終端より手前は切り上げ
        b = max(b, a)  # b>=a に

        if b - a <= min_len:
            # 取り分なし → 直前スロットへ併合（無ければ次のスロットへ）
            if merged_groups:
                merged_groups[-1].extend(group)
            else:
                pending.extend(group)
            continue

        merged_groups.append(pending + list(group))
        pending = []
        segs.append([a, b])
        cur = b

    if pending and merged_groups:
        merged_groups[-1].extend(pending)

    # 末尾が hi に届かないなら、最後を伸ばす
    if segs:
        segs[-1][1] = hi

    return segs, merged_groups


def normalize_row_with_out_edges(
    dag: nx.DiGraph,
    head: str,
    dir_key: str = "dir",
    out_edge_type: str = "R",
    in_edge_type: str = "L",
    thresholds=None,
) -> nx.DiGraph:
    """
    head から辿れるノードを対象に、横方向(out_edge_type/in_edge_type)で
    out が複数あるノードを分割して 1:1 化する。

    - out_edge_type="R" のとき: 右方向に分割（従来）
    - out_edge_type="L" のとき: 左方向に分割（左右反転対応）
    """
    if thresholds is None:
        thresholds = _calc_adjacency_thresholds(
            [dag.nodes[n]["bbox"] for n in dag.nodes]
        )

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
                    G.nodes[bwd]["bbox"],
                    G.nodes[dup]["bbox"],
                    rule="soft",
                    dist_threshold=thresholds["dist"],
                    ignore_dist_threshold=thresholds["ignore"],
                    overlap_ratio_th=0.25,
                )

        elif out_edge_type == "L":
            # 右側セル（右→u が L）
            outs_bwd = [p for p in G.predecessors(u) if G[p][u].get(dir_key) == "L"]

            def ok_adj(bwd, dup):
                # bwd(右) は dup の右にいるはず（判定向きを反転）
                return is_right_adjacent(
                    G.nodes[dup]["bbox"],
                    G.nodes[bwd]["bbox"],
                    rule="soft",
                    dist_threshold=thresholds["dist"],
                    ignore_dist_threshold=thresholds["ignore"],
                    overlap_ratio_th=0.25,
                )

        else:
            outs_bwd = []

            def ok_adj(bwd, dup):
                return False

        # 縦方向（D/U）は常に同じ定義：上→下がD、下→上がU
        up_cells = [p for p in G.predecessors(u) if G[p][u].get(dir_key) == "D"]  # 上→u
        down_cells = [v for v in G.successors(u) if G[u][v].get(dir_key) == "D"]  # u→下

        # ほぼ同一の y区間を持つ out（重複検出セル）は同一スロットにまとめ、
        # 取り分が空になるグループ（包含された重複検出）は直前スロットへ併合
        out_groups = []
        seg_spans = []
        if len(outs_fwd) > 1:
            groups, group_intervals = _group_outs_by_axis_interval(
                G, outs_fwd, axis=1
            )
            _, uy1, _, uy2 = G.nodes[u]["bbox"]
            # 最小セル寸法よりはるかに細いスロットは実在の行ではない
            seg_spans, out_groups = _assign_slots_by_intervals(
                uy1, uy2, groups, group_intervals, min_len=thresholds["ignore"]
            )

        if len(out_groups) > 1:
            # スロットの y区間（中心順・上→下に単調補正）で u を縦分割する。
            # 分割区間とグループは同じ順序で構築されるため、zip のペアリングが
            # ねじれない（y1 順で並べ直すと、y1 が同値だったり区間が重複し合う
            # 隣接セルで対応がねじれ、斜めエッジが生じる）。
            ux1, uy1, ux2, uy2 = G.nodes[u]["bbox"]
            seg_bboxes = [(ux1, a, ux2, b) for a, b in seg_spans]

            base_attr = dict(G.nodes[u])
            dups = []

            # 各分割 bbox ごとに新ノードを作成（衝突しない名前）
            for bb in seg_bboxes:
                nu = f"{u}__dup{next(dup_counter)}"
                attr = dict(base_attr)
                attr["bbox"] = bb
                G.add_node(nu, **attr)
                dups.append(nu)

            # 前方(out) グループと複製ノードを 1:1 で接続
            # （グループ内の重複セルはすべて同じ複製ノードへ接続する）
            for group, dup in zip(out_groups, dups):
                for out in group:
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
    thresholds=None,
) -> nx.DiGraph:
    """
    head から辿れるノードを対象に、右隣セルが複数あるノードを分割して 1:1 化する。
    """
    if thresholds is None:
        thresholds = _calc_adjacency_thresholds(
            [dag.nodes[n]["bbox"] for n in dag.nodes]
        )

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
                    G.nodes[bwd]["bbox"],
                    G.nodes[dup]["bbox"],
                    rule="soft",
                    dist_threshold=thresholds["dist"],
                    ignore_dist_threshold=thresholds["ignore"],
                    overlap_ratio_th=0.25,
                )
        elif out_edge_type == "U":
            outs_bwd = [v for v in G.successors(u) if G[u][v].get(dir_key) == "D"]

            def ok_adj(bwd, dup):
                return is_bottom_adjacent(
                    G.nodes[dup]["bbox"],
                    G.nodes[bwd]["bbox"],
                    rule="soft",
                    dist_threshold=thresholds["dist"],
                    ignore_dist_threshold=thresholds["ignore"],
                    overlap_ratio_th=0.25,
                )
        else:
            outs_bwd = []

            def ok_adj(bwd, dup):
                return False

        left_cells = [p for p in G.predecessors(u) if G[p][u].get(dir_key) == "R"]
        right_cells = [v for v in G.successors(u) if G[u][v].get(dir_key) == "R"]
        # ほぼ同一の x区間を持つ out（重複検出セル）は同一スロットにまとめ、
        # 取り分が空になるグループ（包含された重複検出）は直前スロットへ併合
        out_groups = []
        seg_spans = []
        if len(outs_fwd) > 1:
            groups, group_intervals = _group_outs_by_axis_interval(
                G, outs_fwd, axis=0
            )
            ux1, _, ux2, _ = G.nodes[u]["bbox"]
            # 最小セル寸法よりはるかに細いスロットは実在の列ではない
            seg_spans, out_groups = _assign_slots_by_intervals(
                ux1, ux2, groups, group_intervals, min_len=thresholds["ignore"]
            )

        if len(out_groups) > 1:
            # スロットの x区間（中心順・左→右に単調補正）で u を横分割する。
            # 分割区間とグループは同じ順序で構築されるため、zip のペアリングが
            # ねじれない（x1 順で並べ直すと、x1 が同値だったり区間が重複し合う
            # 隣接セルで対応がねじれ、斜めエッジが生じる）。
            ux1, uy1, ux2, uy2 = G.nodes[u]["bbox"]
            seg_bboxes = [(a, uy1, b, uy2) for a, b in seg_spans]

            base_attr = dict(G.nodes[u])
            dups = []

            # 各分割 bbox ごとに新ノードを作成
            for bb in seg_bboxes:
                nu = f"{u}__dup{next(dup_counter)}"
                attr = dict(base_attr)
                attr["bbox"] = bb
                G.add_node(nu, **attr)
                dups.append(nu)

            # 前方(out) グループと複製ノードを 1:1 で接続
            # （グループ内の重複セルはすべて同じ複製ノードへ接続する）
            for group, dup in zip(out_groups, dups):
                for out in group:
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

            for dup in dups:
                queue.append(dup)

            # 元ノード削除
            G.remove_node(u)
        else:
            for v in outs_fwd:
                queue.append(v)

    return G


def expand_dir_to_uit_row(
    dag: nx.DiGraph,
    dir_key: str = "dir",
    thresholds=None,
) -> nx.DiGraph:
    """
    target_dir（クラスタ抽出に使う方向）で線クラスタの head を取り、
    その head から edge_fwd/edge_bwd のペアを 1:1 化する。
    """
    G = dag.copy()

    line_heads, line_clusters = _cluster_heads_by_in_degree(G, dir_value="R")
    for head in line_heads:
        G = normalize_row_with_out_edges(
            G,
            head,
            dir_key=dir_key,
            in_edge_type="L",
            out_edge_type="R",
            thresholds=thresholds,
        )

    line_heads, line_clusters = _cluster_heads_by_in_degree(G, dir_value="L")
    for head in line_heads:
        G = normalize_row_with_out_edges(
            G,
            head,
            dir_key=dir_key,
            in_edge_type="R",
            out_edge_type="L",
            thresholds=thresholds,
        )

    return G


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


def _cluster_heads_by_in_degree(dag: nx.DiGraph, dir_value: str):
    """ヘッドから隣接ノードへ辿るクラスタリングを行う。"""

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
            heads.append(min(comp))

    return heads, comps


def expand_dir_to_uit_col(
    dag: nx.DiGraph,
    dir_key: str = "dir",
    thresholds=None,
) -> nx.DiGraph:
    """
    target_dir（クラスタ抽出に使う方向）で線クラスタの head を取り、
    その head から edge_fwd/edge_bwd のペアを 1:1 化する。
    """
    G = dag.copy()

    line_heads, line_clusters = _cluster_heads_by_in_degree(G, dir_value="D")
    for head in line_heads:
        G = normalize_col_with_out_edges(
            G,
            head,
            dir_key=dir_key,
            in_edge_type="U",
            out_edge_type="D",
            thresholds=thresholds,
        )

    line_heads, line_clusters = _cluster_heads_by_in_degree(G, dir_value="U")
    for head in line_heads:
        G = normalize_col_with_out_edges(
            G,
            head,
            dir_key=dir_key,
            in_edge_type="D",
            out_edge_type="U",
            thresholds=thresholds,
        )

    return G


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


def _expand_grid_to_unit(dag: nx.DiGraph, thresholds=None) -> nx.DiGraph:
    """dagをユニットセル化（行・列方向に分割して1:1化）する。"""

    dag = expand_dir_to_uit_row(dag, thresholds=thresholds)
    dag = expand_dir_to_uit_col(dag, thresholds=thresholds)
    return dag


def _get_grid_from_dag(dag: nx.DiGraph) -> List[List[str]]:
    """dagから行列形式のセル配置を取得する。"""

    row_root = _cluster_heads_by_in_degree(dag, dir_value="R")[0]
    col_root = _cluster_heads_by_in_degree(dag, dir_value="D")[0]

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
        row_nodes = get_line_with_head(dag, r_head, dir_value="R")
        row_nodes = sorted(
            row_nodes,
            key=lambda n: dag.nodes[n]["bbox"][0],
        )

        row = []
        for c_head in col_root:
            c_nodes = get_line_with_head(dag, c_head, dir_value="D")
            c_nodes = sorted(
                c_nodes,
                key=lambda n: dag.nodes[n]["bbox"][1],
            )
            cell_id = set(row_nodes).intersection(set(c_nodes))
            if cell_id:
                row.append(list(cell_id)[0])
            else:
                row.append(None)
        raw_data.append(row)

    return raw_data


def _assign_cell_positions(cells, data):
    """セルの row/col/row_span/col_span を data から計算して設定する。"""
    cell_info = _calc_spans_and_indices_from_raw_grid(data)
    for cell_id, info in cell_info.items():
        cell = cells[cell_id]
        cell.row = info["row"]
        cell.col = info["col"]
        cell.row_span = info["row_span"]
        cell.col_span = info["col_span"]


def _remove_dup_suffix_from_data(grid):
    """data 内の cell_id から __dup... を削除する。"""
    data = []
    for i, row in enumerate(grid):
        data.append(
            [
                cell_id.split("__dup")[0] if cell_id is not None else None
                for cell_id in row
            ]
        )
    return data


def _get_col_headers_from_grid(grid, is_header_row, cells, clustered_nodes):
    n_cols = len(grid[0])
    n_rows = len(grid)

    header_ids = set()
    col_headers = []
    for col_idx in range(n_cols):
        col_header = []
        for row_idx in range(n_rows):
            if is_header_row[row_idx]:
                if grid[row_idx][col_idx] is not None:
                    col_header.append(grid[row_idx][col_idx])
                    header_ids.add(grid[row_idx][col_idx])

        col_header = list(set(col_header))

        col_header = sorted(
            col_header,
            key=lambda h: cells[h].box[1],
        )

        col_headers.append(col_header)

    gird_cells = set()
    for cluster in clustered_nodes.values():
        for cell in cluster:
            gird_cells.add(cell.id)

    for cell in cells.values():
        if (
            cell.id not in header_ids
            and cell.role == "header"
            and cell.id in gird_cells
        ):
            cell.role = "cell"

    return col_headers


def _get_grid_bbox(grid, cells) -> BBox:
    """grid 全体の bbox を取得する。"""
    x1_list = []
    y1_list = []
    x2_list = []
    y2_list = []

    for row in grid:
        for cell_id in row:
            if cell_id is None:
                continue
            cell = cells[cell_id]
            x1, y1, x2, y2 = cell.box
            x1_list.append(x1)
            y1_list.append(y1)
            x2_list.append(x2)
            y2_list.append(y2)

    if not x1_list or not y1_list or not x2_list or not y2_list:
        return (0.0, 0.0, 0.0, 0.0)

    return (min(x1_list), min(y1_list), max(x2_list), max(y2_list))


def merge_cells(cell1, cell2):
    # すでに片方のIDがもう片方に含まれている場合、大きい方を返す
    cell1_ids = set(cell1.id.split("_"))
    cell2_ids = set(cell2.id.split("_"))

    if cell1_ids.issubset(cell2_ids):
        return cell2, False

    if cell2_ids.issubset(cell1_ids):
        return cell1, False

    """2つのセルをマージして新しいセルを返す。"""
    x1_1, y1_1, x2_1, y2_1 = cell1.box
    x1_2, y1_2, x2_2, y2_2 = cell2.box

    new_x1 = min(x1_1, x1_2)
    new_y1 = min(y1_1, y1_2)
    new_x2 = max(x2_1, x2_2)
    new_y2 = max(y2_1, y2_2)

    new_contents = cell1.contents + cell2.contents

    new_cell = type(cell1)(
        id=f"{cell1.id}_{cell2.id}",
        box=(new_x1, new_y1, new_x2, new_y2),
        role=cell1.role,
        contents=new_contents.strip(),
        row=min(cell1.row, cell2.row),
        col=min(cell1.col, cell2.col),
        row_span=cell1.row_span + cell2.row_span,
        col_span=cell1.col_span + cell2.col_span,
    )
    return new_cell, True


def _merge_same_column_values(grid, col_headers, cells):
    union_find = UnionFind(len(col_headers))

    for c1 in range(len(col_headers)):
        if not col_headers[c1]:
            continue

        for c2 in range(c1 + 1, len(col_headers)):
            if not col_headers[c2]:
                continue

            if col_headers[c1][-1] == col_headers[c2][-1]:
                union_find.union(c1, c2)

    new_col_headers = []
    for group in union_find.groups():
        merged = []
        for c in group:
            merged.extend(col_headers[c])

        merged = list(set(merged))
        merged = sorted(merged, key=lambda h: cells[h].box[1])
        new_col_headers.append(list(set(merged)))

    new_grid = []
    for row in grid:
        new_row = []
        for group in union_find.groups():
            cell_ids = [row[c] for c in group if row[c] is not None]

            if not cell_ids:
                new_row.append(None)
            else:
                merged_cell = cells[cell_ids[0]]

                for cid in cell_ids[1:]:
                    merged_cell, is_merged = merge_cells(merged_cell, cells[cid])

                new_row.append(merged_cell.id)
                cells[merged_cell.id] = merged_cell

        new_grid.append(new_row)

    grid_cells = set()
    for row in new_grid:
        for cell_id in row:
            if cell_id is not None:
                grid_cells.add(cell_id)

    cells = {cid: cell for cid, cell in cells.items() if cid in grid_cells}

    return new_grid, new_col_headers, cells


def parse_grid_from_bottom_up(cells, clustered_nodes, merge_same_column_values=False):
    # クラスタ内セルの寸法統計から隣接判定しきい値を算出し、
    # DAG構築とユニットセル展開で同一のしきい値を用いる
    grid_nodes = (
        clustered_nodes["cell"] + clustered_nodes["empty"] + clustered_nodes["header"]
    )
    thresholds = _calc_adjacency_thresholds([c.box for c in grid_nodes])

    dag = _get_grid_dag(clustered_nodes, thresholds=thresholds)
    dag = _expand_grid_to_unit(dag, thresholds=thresholds)

    grid = _get_grid_from_dag(dag)

    if len(grid) == 0 or len(grid[0]) == 0:
        return None

    roles = [
        [
            dag.nodes[cell_id]["role"] if cell_id is not None else "empty"
            for cell_id in row
        ]
        for row in grid
    ]

    is_header_row = [
        all(role == "header" or role == "empty" for role in row) for row in roles
    ]

    grid = _remove_dup_suffix_from_data(grid)
    grid_box = list(map(int, _get_grid_bbox(grid, cells)))
    _assign_cell_positions(cells, grid)
    col_headers = _get_col_headers_from_grid(
        grid, is_header_row, cells, clustered_nodes
    )

    if merge_same_column_values:
        grid, col_headers, cells = _merge_same_column_values(grid, col_headers, cells)

    return (
        TableGridSchema(
            id=None,
            n_row=len(grid),
            n_col=len(grid[0]) if grid else 0,
            box=grid_box,
            data=grid,
            col_headers=col_headers,
        ),
        cells,
        dag,
    )
