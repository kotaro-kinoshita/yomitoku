import math
import cv2

import asyncio
from concurrent.futures import ThreadPoolExecutor

from .table_detector import TableDetector
from .cell_detector import CellDetector
from .text_detector import TextDetector
from .text_recognizer import TextRecognizer
from .document_analyzer import prediction_reading_order, combine_flags
from .ocr import OCRSchema, ocr_aggregate


from .utils.misc import calc_overlap_ratio, is_contained, quad_to_xyxy
from .utils.union_find import UnionFind
from .utils.visualizer import det_visualizer
from .schemas import ParagraphSchema

from collections import defaultdict

import networkx as nx


def clamp(t, lo, hi):
    return max(lo, min(hi, t))


def point_to_segment_distance(px, py, ax, ay, bx, by):
    """
    点(px,py)と線分(ax,ay)-(bx,by)の最短距離を計算
    """

    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    denom = abx * abx + aby * aby
    if denom == 0:
        return math.hypot(px - ax, py - ay)
    t = (apx * abx + apy * aby) / denom
    t = clamp(t, 0.0, 1.0)
    cx, cy = ax + t * abx, ay + t * aby
    return math.hypot(px - cx, py - cy)


def right_edge_to_left_edge_dist(A, B):
    """
    Aの右辺とBの左辺の最短距離を計算
    """

    ax1, ay1, ax2, ay2 = A  # A right edge: (ax2,ay1)-(ax2,ay2)
    bx1, by1, bx2, by2 = B  # B left  edge: (bx1,by1)-(bx1,by2)

    # Aの右上 -> B左辺
    d1 = point_to_segment_distance(ax2, ay1, bx1, by1, bx1, by2)

    # Aの右下 -> B左辺
    d2 = point_to_segment_distance(ax2, ay2, bx1, by1, bx1, by2)

    # Bの左上 -> A右辺
    d3 = point_to_segment_distance(bx1, by1, ax2, ay1, ax2, ay2)

    # Bの左下 -> A右辺
    d4 = point_to_segment_distance(bx1, by2, ax2, ay1, ax2, ay2)

    return max(d1, d4), max(d2, d3), max(d3, d4), max(d1, d2)


def top_edge_to_bottom_edge_dist(A, B):
    """
    Aの下辺とBの上辺の最短距離を計算
    """

    ax1, ay1, ax2, ay2 = A  # A bottom edge: (ax1,ay2)-(ax2,ay2)
    bx1, by1, bx2, by2 = B  # B top    edge: (bx1,by1)-(bx2,by1)

    # Aの左下 -> B上辺
    d1 = point_to_segment_distance(ax1, ay2, bx1, by1, bx2, by1)

    # Aの右下 -> B上辺
    d2 = point_to_segment_distance(ax2, ay2, bx1, by1, bx2, by1)

    # Bの左上 -> A下辺
    d3 = point_to_segment_distance(bx1, by1, ax1, ay2, ax2, ay2)

    # Bの右上 -> A下辺
    d4 = point_to_segment_distance(bx2, by1, ax1, ay2, ax2, ay2)

    return max(d1, d4), max(d2, d3), max(d3, d4), max(d1, d2)


def overlap_interval(i1, i2, j1, j2):
    """
    [i1,i2] と [j1,j2] の重なり長
    """
    return max(0.0, min(i2, j2) - max(i1, j1))


def point_distance(p, q):
    px, py = p
    qx, qy = q
    return math.hypot(px - qx, py - qy)


def gap_interval(interval_a, interval_b):
    """
    interval_a = (a1, a2)
    interval_b = (b1, b2)
    2区間の最短距離（重なれば0）
    """
    a1, a2 = interval_a
    b1, b2 = interval_b

    if b2 < a1:
        return a1 - b2
    if a2 < b1:
        return b1 - a2
    return 0.0


def is_right_adjacent(box_a, box_b, threshold=10, overlap_ratio_th=0.5, rule="soft"):
    """
    box_aの右隣にbox_bがあるか判定
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # 1) 方向制約：BがAの右側（最低条件）
    if bx1 < ax1:
        return False

    # 2) 縦方向の重なり制約
    if overlap_interval(ay1, ay2, by1, by2) < overlap_ratio_th * min(
        ay2 - ay1, by2 - by1
    ):
        return False

    # 3) 頂点間の距離の制約
    # Aの右下とBの左上が近すぎる場合は隣接しない
    if point_distance((ax2, ay2), (bx1, by1)) < threshold:
        return False

    # Aの右上とBの左下が近すぎる場合は隣接しない
    if point_distance((ax2, ay1), (bx1, by2)) < threshold:
        return False

    # 4) 頂点と線分距離制約
    d1, d2, d3, d4 = right_edge_to_left_edge_dist(box_a, box_b)
    if rule == "hard":
        if d1 < threshold and d2 < threshold and d3 < threshold and d4 < threshold:
            return True
    elif rule == "soft":
        if d1 < threshold or d2 < threshold or d3 < threshold or d4 < threshold:
            return True

    return False


def is_bottom_adjacent(box_a, box_b, threshold=10, overlap_ratio_th=0.5, rule="soft"):
    """
    box_aの下隣にbox_bがあるか判定
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # 1) 方向制約：BがAの下側（最低条件）
    if by1 < ay1:
        return False

    # 2) 横方向の重なり制約
    if overlap_interval(ax1, ax2, bx1, bx2) < overlap_ratio_th * min(
        ax2 - ax1, bx2 - bx1
    ):
        return False

    # 3) 頂点間の距離の制約
    # Aの右下とBの左上が近すぎる場合は隣接しない
    if point_distance((ax2, ay2), (bx1, by1)) < threshold:
        return False

    # Aの左下とBの右上が近すぎる場合は隣接しない
    if point_distance((ax1, ay2), (bx2, by1)) < threshold:
        return False

    # 4) 頂点間距離制約
    d1, d2, d3, d4 = top_edge_to_bottom_edge_dist(box_a, box_b)

    if rule == "hard":
        if d1 < threshold and d2 < threshold and d3 < threshold and d4 < threshold:
            return True
    elif rule == "soft":
        if d1 < threshold or d2 < threshold or d3 < threshold or d4 < threshold:
            return True

    return False


def extract_words_within_element(pred_words, element):
    contained_words = []
    word_sum_width = 0
    word_sum_height = 0
    check_list = [False] * len(pred_words)

    for i, word in enumerate(pred_words):
        word_box = quad_to_xyxy(word.points)
        if is_contained(element["box"], word_box, threshold=0.5):
            word_sum_width += word_box[2] - word_box[0]
            word_sum_height += word_box[3] - word_box[1]
            check_list[i] = True

            word_element = ParagraphSchema(
                box=word_box,
                contents=word.content,
                direction=word.direction,
                order=0,
                role=None,
            )
            contained_words.append(word_element)

    if len(contained_words) == 0:
        return None, None, check_list

    word_direction = [word.direction for word in contained_words]
    cnt_horizontal = word_direction.count("horizontal")
    cnt_vertical = word_direction.count("vertical")

    element_direction = "horizontal" if cnt_horizontal > cnt_vertical else "vertical"
    order = "left2right" if element_direction == "horizontal" else "right2left"
    prediction_reading_order(contained_words, order)
    contained_words = sorted(contained_words, key=lambda x: x.order)

    contained_words = "\n".join([content.contents for content in contained_words])

    return (contained_words, element_direction, check_list)


def walk_to_terminal_by_edge_type_bbox_axis(
    G,
    start,
    edge_attr: str,
    edge_value,
    bbox_attr: str = "bbox",
    axis: str = "x",  # "x" or "y"
    use: str = "center",  # "center" or "min" or "max"
    max_steps: int = 10_000,
):
    """
    特定タイプの out edge だけ辿る。
    分岐したら bbox の x もしくは y の差が最小の次ノードを選ぶ（greedy）。
    そのタイプの出エッジが無くなったら終了して path を返す。
    """
    assert axis in ("x", "y")
    assert use in ("center", "min", "max")

    def axis_value(node):
        bbox = G.nodes[node].get(bbox_attr)
        if bbox is None:
            raise ValueError(f"Node {node} lacks '{bbox_attr}'")
        x0, y0, x1, y1 = bbox
        if axis == "x":
            return (x0 + x1) / 2 if use == "center" else (x0 if use == "min" else x1)
        else:
            return (y0 + y1) / 2 if use == "center" else (y0 if use == "min" else y1)

    path = [start]
    cur = start
    seen = {start}

    for _ in range(max_steps):
        cur_val = axis_value(cur)

        candidates = []
        for _, v, data in G.out_edges(cur, data=True):
            if data.get(edge_attr) != edge_value:
                continue
            v_val = axis_value(v)
            dist = abs(v_val - cur_val)  # ←「近い」の定義：x or y の差だけ
            candidates.append((dist, v))

        # 終端（このタイプの out edge が無い）
        if not candidates:
            break

        candidates.sort(key=lambda t: t[0])
        nxt = candidates[0][1]

        # サイクル対策（必要ならここは例外にする等に変更）
        if nxt in seen:
            path.append(nxt)
            break

        path.append(nxt)
        seen.add(nxt)
        cur = nxt

    return path


def make_unique_all(seq):
    counter = defaultdict(int)
    result = []

    for x in seq:
        result.append(f"{x}_{counter[x]}")
        counter[x] += 1

    return result


class TableSemanticParser:
    def __init__(self, configs={}, device="cuda:1", visualize=False):
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
        self.table_structure_recognizer = CellDetector(
            **table_parser_kwargs,
        )

        self.text_detector = TextDetector(
            **text_detector_kwargs,
        )

        self.text_recognizer = TextRecognizer(
            **text_recognizer_kwargs,
        )

        self.visualize = visualize

    def _matching_group_and_cells(self, nodes):
        match = {
            "header_to_group": {},
            "group_to_cells": {},
            "cell_to_groups": {},
            "group_to_header": {},
        }

        if len(nodes["group"]) == 0:
            return match

        B = nx.Graph()
        U = [header["id"] for header in nodes["header"]]
        V = [group["id"] for group in nodes["group"]]

        B.add_nodes_from(U, bipartite=0)
        B.add_nodes_from(V, bipartite=1)

        # ヘッダーとグループのマッチング
        for header in nodes["header"]:
            for i, group in enumerate(nodes["group"]):
                ratio = calc_overlap_ratio(header["box"], group["box"])
                if ratio[0] > 0.0:
                    B.add_edge(header["id"], group["id"], weight=ratio[0])

        # 最大重みマッチングを計算
        pairs = nx.algorithms.matching.max_weight_matching(
            B, maxcardinality=True, weight="weight"
        )

        matched_header_to_group = {}
        for a, b in pairs:
            if a in U:
                matched_header_to_group[a] = b
            else:
                matched_header_to_group[b] = a

        # cellsとグループのマッチング
        matched_group_to_cells = {}
        matched_cell_to_groups = {}
        for group in nodes["group"]:
            for cell in nodes["cell"] + nodes["empty"]:
                if is_contained(
                    group["box"],
                    cell["box"],
                    threshold=0.5,
                ):
                    matched_group_to_cells.setdefault(group["id"], []).append(
                        cell["id"]
                    )

                    if cell["id"] not in matched_cell_to_groups:
                        matched_cell_to_groups[cell["id"]] = []

                    matched_cell_to_groups[cell["id"]].append(group["id"])

        match["cell_to_groups"] = matched_cell_to_groups
        match["group_to_cells"] = matched_group_to_cells
        match["header_to_group"] = matched_header_to_group
        match["group_to_header"] = {v: k for k, v in matched_header_to_group.items()}

        return match

    def _calc_adjacent_header_to_header(self, dag, match, nodes, group_direction):
        """
        ヘッダーの隣接関係を計算
        グループの方向情報を利用して隣接関係を追加
        例えば、あるグループが水平（H）方向であれば、そのグループに属するヘッダー同士は水平方向の隣接関係を持つ可能性があると判断する
        """

        header_to_group = match["header_to_group"]

        for node in nodes:
            for potential_parent in nodes:
                if node["id"] == potential_parent["id"]:
                    continue

                potential_parent_group_id = header_to_group.get(
                    potential_parent["id"], None
                )
                node_group_id = header_to_group.get(node["id"], None)

                if (
                    group_direction.get(potential_parent_group_id, None) == "H"
                    or group_direction.get(node_group_id, None) == "H"
                ):
                    # 左右の隣接判定
                    if is_right_adjacent(potential_parent["box"], node["box"]):
                        dag.add_edge(potential_parent["id"], node["id"], dir="R")
                        dag.add_edge(node["id"], potential_parent["id"], dir="L")

                if (
                    group_direction.get(potential_parent_group_id, None) == "V"
                    or group_direction.get(node_group_id, None) == "V"
                ):
                    # 上下の隣接判定
                    if is_bottom_adjacent(potential_parent["box"], node["box"]):
                        dag.add_edge(potential_parent["id"], node["id"], dir="D")
                        dag.add_edge(node["id"], potential_parent["id"], dir="U")

    def _calc_adjacent_cell_to_cell(self, dag, match, nodes):
        """
        セルの隣接関係を計算
        セルが同じサブテーブル or 同じグループに属している場合のみ隣接関係を追加
        """

        cell_to_sub_table = match["cell_to_sub_table"]
        cell_to_groups = match["cell_to_groups"]

        for node in nodes:
            for potential_parent in nodes:
                if node["id"] == potential_parent["id"]:
                    continue

                node_group_id = cell_to_sub_table.get(node["id"], None)
                potential_parent_id = cell_to_sub_table.get(
                    potential_parent["id"], None
                )

                if node_group_id is None or potential_parent_id is None:
                    node_group_id = cell_to_groups.get(node["id"], [])
                    potential_parent_id = cell_to_groups.get(potential_parent["id"], [])

                    if node_group_id is None or potential_parent_id is None:
                        continue

                    if set(node_group_id) != set(potential_parent_id):
                        continue

                elif node_group_id != potential_parent_id:
                    continue

                # 左右の隣接判定
                if is_right_adjacent(potential_parent["box"], node["box"]):
                    dag.add_edge(potential_parent["id"], node["id"], dir="R")
                    dag.add_edge(node["id"], potential_parent["id"], dir="L")

                # 上下の隣接判定
                if is_bottom_adjacent(potential_parent["box"], node["box"]):
                    dag.add_edge(potential_parent["id"], node["id"], dir="D")
                    dag.add_edge(node["id"], potential_parent["id"], dir="U")

    def aggregate(self, ocr_res, cells):
        check_list = [False] * len(ocr_res.words)
        for cell in cells:
            words, direction, flags = extract_words_within_element(ocr_res.words, cell)

            if words is None:
                words = ""

            cell["contents"] = words
            check_list = combine_flags(check_list, flags)

    def _split_nodes_with_role(self, table):
        nodes = {
            "header": [],
            "group": [],
            "cell": [],
            "empty": [],
        }
        for cell in table["cells"]:
            if cell["role"] not in nodes:
                nodes[cell["role"]] = []
            nodes[cell["role"]].append(cell)

        return nodes

    def _calc_adjacent_header_to_cell(self, dag, match, headers, cells):
        """
        グループ内のヘッダーとセルの隣接関係を計算
        """

        cell_groups = match["cell_to_groups"]
        header_groups = match["header_to_group"]

        group_direction = {}
        for header in headers:
            for cell in cells:
                if is_right_adjacent(header["box"], cell["box"]):
                    header_group_id = header_groups.get(header["id"], None)
                    cell_group_ids = cell_groups.get(cell["id"], None)

                    if header_group_id is None or cell_group_ids is None:
                        continue

                    if header_group_id in cell_group_ids:
                        dag.add_edge(header["id"], cell["id"], dir="R")
                        dag.add_edge(cell["id"], header["id"], dir="L")
                        group_direction[header_group_id] = "H"

                if is_bottom_adjacent(header["box"], cell["box"]):
                    header_group_id = header_groups.get(header["id"], None)
                    cell_group_ids = cell_groups.get(cell["id"], None)

                    if header_group_id is None or cell_group_ids is None:
                        continue

                    if header_group_id in cell_group_ids:
                        dag.add_edge(header["id"], cell["id"], dir="D")
                        dag.add_edge(cell["id"], header["id"], dir="U")
                        group_direction[header_group_id] = "V"

        return group_direction

    def _get_pair_groups(self, nodes, group_direction):
        rows = [
            group
            for group in nodes["group"]
            if group["id"] in group_direction and group_direction[group["id"]] == "H"
        ]
        return rows

    def _get_sub_tables(self, match, nodes, group_direction):
        """
        グループの方向情報を利用して、列をマージし、サブテーブルを生成
        """

        cols = [
            group
            for group in nodes["group"]
            if group["id"] in group_direction and group_direction[group["id"]] == "V"
        ]

        group_cells = match["group_to_cells"]

        union_find = UnionFind(len(cols))

        for i, col_a in enumerate(cols):
            if col_a["id"] not in group_cells:
                continue

            for j, col_b in enumerate(cols):
                if col_a["id"] == col_b["id"]:
                    continue

                if is_right_adjacent(col_a["box"], col_b["box"], rule="soft"):
                    union_find.union(i, j)

        sub_tables = []
        for groups in union_find.groups():
            x1 = min([cols[i]["box"][0] for i in groups])
            y1 = min([cols[i]["box"][1] for i in groups])
            x2 = max([cols[i]["box"][2] for i in groups])
            y2 = max([cols[i]["box"][3] for i in groups])

            sub_tables.append(
                {
                    "id": len(sub_tables) + 1,
                    "box": [x1, y1, x2, y2],
                    "member_ids": [cols[i]["id"] for i in groups],
                }
            )

        return sub_tables

    def _matching_cells_within_sub_tables(self, match, sub_tables):
        matched_cell_and_sub_table = {}
        group_to_cells = match["group_to_cells"]

        for sub_table in sub_tables:
            member_group_ids = sub_table["member_ids"]

            for group_id in member_group_ids:
                if group_id not in group_to_cells:
                    continue

                for cell_id in group_to_cells[group_id]:
                    matched_cell_and_sub_table[cell_id] = sub_table["id"]

        match["cell_to_sub_table"] = matched_cell_and_sub_table

    def _get_sub_table_n_rows(self, match, sub_table, dag):
        n_rows = 0
        base_cols = []

        group_to_header = match["group_to_header"]
        for group_id in sub_table["member_ids"]:
            header_id = group_to_header.get(group_id, None)
            nodes = walk_to_terminal_by_edge_type_bbox_axis(
                dag,
                header_id,
                edge_attr="dir",
                edge_value="D",
                axis="y",
                use="center",
            )

            cell_nodes = [
                node for node in nodes if dag.nodes[node]["role"] in ["cell", "empty"]
            ]

            if n_rows < len(cell_nodes):
                base_cols = cell_nodes
                n_rows = len(cell_nodes)

        return n_rows, base_cols

    def _calc_pairs(self, dag, match, groups):
        group_to_cells = match["group_to_cells"]
        pairs = []
        for group in groups:
            cells = group_to_cells.get(group["id"], None)
            for cell in cells:
                is_not_reef = any(
                    data.get("dir") == "R"
                    for _, _, data in dag.out_edges(cell, data=True)
                )

                if is_not_reef:
                    continue

                search_left = walk_to_terminal_by_edge_type_bbox_axis(
                    dag,
                    cell,
                    edge_attr="dir",
                    edge_value="L",
                    axis="y",
                    use="center",
                )

                key = []
                value = []
                for node in search_left:
                    if dag.nodes[node]["role"] in ["cell", "empty"]:
                        value.append(dag.nodes[node])

                    elif dag.nodes[node]["role"] == "header":
                        key.append(dag.nodes[node])

                key = sorted(key, key=lambda x: x["bbox"][0])
                value = sorted(value, key=lambda x: x["bbox"][0])

                pairs.append(
                    {
                        "key": [k["id"] for k in key],
                        "value": [v["id"] for v in value],
                    }
                )

        return pairs

    def _calc_matrix(
        self,
        dag,
        match,
        sub_tables,
    ):
        matrices = []
        for j, sub_table in enumerate(sub_tables):
            # 最も行数の多い列を基準列として取得
            n_rows, base_cols = self._get_sub_table_n_rows(match, sub_table, dag)
            base_cols = sorted(base_cols, key=lambda x: dag.nodes[x]["bbox"][1])

            rows = []
            for cell in base_cols:
                # 基準列のセルから左右に辿って行全体を取得
                search_right = walk_to_terminal_by_edge_type_bbox_axis(
                    dag,
                    cell,
                    edge_attr="dir",
                    edge_value="R",
                    axis="y",
                    use="center",
                )

                search_left = walk_to_terminal_by_edge_type_bbox_axis(
                    dag,
                    cell,
                    edge_attr="dir",
                    edge_value="L",
                    axis="y",
                    use="center",
                )
                row_cells = set(search_right) | set(search_left)
                row_cells = sorted(row_cells, key=lambda x: dag.nodes[x]["bbox"][0])

                row_header = [
                    node for node in row_cells if dag.nodes[node]["role"] == "header"
                ]

                row_value = [
                    node
                    for node in row_cells
                    if dag.nodes[node]["role"] in ["cell", "empty"]
                ]

                col_headers = []

                # 各セルから上方向に辿って列ヘッダーを取得
                for value in row_value:
                    search_top = walk_to_terminal_by_edge_type_bbox_axis(
                        dag,
                        value,
                        edge_attr="dir",
                        edge_value="U",
                        axis="x",
                        use="center",
                    )

                    col_header = [
                        node
                        for node in search_top
                        if dag.nodes[node]["role"] == "header"
                    ]

                    col_headers.append(
                        sorted(
                            col_header,
                            key=lambda x: dag.nodes[x]["bbox"][1],
                        )
                    )

                row = []
                for col_header, v in zip(col_headers, row_value):
                    col_header = [dag.nodes[h]["id"] for h in col_header]
                    row_header = [dag.nodes[h]["id"] for h in row_header]
                    row.append(
                        {
                            "row_keys": row_header,
                            "col_keys": col_header,
                            "value": dag.nodes[v]["id"],
                        }
                    )

                rows.append(row)
            matrices.append(rows)
        return matrices

    async def run(self, img, id):
        with ThreadPoolExecutor(max_workers=2) as executor:
            loop = asyncio.get_running_loop()
            tasks = [
                loop.run_in_executor(executor, self.text_detector, img),
                loop.run_in_executor(executor, self.layout_parser, img),
            ]

            results = await asyncio.gather(*tasks)

        results_det, _ = results[0]
        results_layout, vis_layout = results[1]

        vis_det = None
        if self.visualize:
            vis_det = det_visualizer(
                img,
                results_det.points,
            )

        table_boxes = [table["box"] for table in results_layout["tables"]]
        results_table, vis_cell, vis_group = self.table_structure_recognizer(
            img, table_boxes, vis=vis_layout
        )

        results_rec, vis_ocr = self.text_recognizer(img, results_det.points, vis_det)
        outputs = {"words": ocr_aggregate(results_det, results_rec)}
        results_ocr = OCRSchema(**outputs)

        return results_ocr, results_table, vis_cell, vis_group, vis_ocr

    def __call__(self, img, id):
        results_ocr, results_table, vis_cell, vis_group, vis_ocr = asyncio.run(
            self.run(img, id)
        )

        semantic_info = []
        for table in results_table:
            self.aggregate(results_ocr, table["cells"])

        for i, table in enumerate(results_table):
            table_information = {
                "cells": {},
                "pair": [],
                "matrix": [],
            }

            nodes = self._split_nodes_with_role(table)
            match = self._matching_group_and_cells(nodes)

            dag = nx.DiGraph()
            for node in nodes["header"] + nodes["cell"] + nodes["empty"]:
                dag.add_node(
                    node["id"],
                    id=node["id"],
                    bbox=node["box"],
                    role=node["role"],
                    contents=node["contents"],
                )

                table_information["cells"][node["id"]] = {
                    "role": node["role"],
                    "bbox": node["box"],
                    "contents": node["contents"],
                }

            group_direction = self._calc_adjacent_header_to_cell(
                dag, match, nodes["header"], nodes["cell"] + nodes["empty"]
            )

            self._calc_adjacent_header_to_header(
                dag, match, nodes["header"], group_direction
            )

            sub_tables = self._get_sub_tables(match, nodes, group_direction)
            pair_groups = self._get_pair_groups(nodes, group_direction)

            self._matching_cells_within_sub_tables(match, sub_tables)
            self._calc_adjacent_cell_to_cell(dag, match, nodes["cell"] + nodes["empty"])

            pairs = self._calc_pairs(
                dag,
                match,
                pair_groups,
            )

            matrices = self._calc_matrix(
                dag,
                match,
                sub_tables,
            )

            table_information["pair"].extend(pairs)
            table_information["matrix"].extend(matrices)
            semantic_info.append(table_information)
            vis_cell = debug_dag(dag, vis_cell, id)

        return semantic_info, vis_cell, vis_group, vis_ocr


def debug_dag(dag, img, id):
    for u, v, attrs in dag.edges(data=True):
        if attrs["dir"] in ["L", "U"]:
            continue
        cx1 = (dag.nodes[u]["bbox"][0] + dag.nodes[u]["bbox"][2]) / 2
        cy1 = (dag.nodes[u]["bbox"][1] + dag.nodes[u]["bbox"][3]) / 2
        cx2 = (dag.nodes[v]["bbox"][0] + dag.nodes[v]["bbox"][2]) / 2
        cy2 = (dag.nodes[v]["bbox"][1] + dag.nodes[v]["bbox"][3]) / 2
        color = (0, 255, 0) if attrs["dir"] == "R" else (255, 0, 0)
        img = cv2.arrowedLine(
            img,
            (int(cx1), int(cy1)),
            (int(cx2), int(cy2)),
            color,
            2,
        )

    return img


def dubug_subtable(i, sub_tables, img):
    for j, sub_table in enumerate(sub_tables):
        x1, y1, x2, y2 = map(int, sub_table["box"])
        img = cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2,
        )
    cv2.imwrite(f"debug/matched_subtables_{id}.png", img)
