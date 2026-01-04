import math

from .table_detector import TableDetector
from .table_parser import TableParser


from .utils.misc import calc_overlap_ratio, is_contained
from .utils.union_find import UnionFind


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


class TableAnalyzer:
    def __init__(self, configs={}, device="cuda", visualize=False):
        table_detector_kwargs = {
            "device": device,
            "visualize": visualize,
        }
        table_parser_kwargs = {
            "device": device,
            "visualize": visualize,
        }

        if isinstance(configs, dict):
            if "table_detector" in configs:
                table_detector_kwargs.update(configs["table_detector"])

            if "table_parser" in configs:
                table_parser_kwargs.update(configs["table_parser"])
        else:
            raise ValueError(
                "configs must be a dict. See the https://kotaro-kinoshita.github.io/yomitoku-dev/usage/"
            )

        self.layout_parser = TableDetector(
            **table_detector_kwargs,
        )
        self.table_structure_recognizer = TableParser(
            **table_parser_kwargs,
        )

    def matching_cells(self, table, nodes):
        matched_group_and_header = {}

        if len(nodes["group"]) == 0:
            return

        # 1) 最大マッチ数 k を取る（重み無視）
        B = nx.Graph()
        U = [header["id"] for header in nodes["header"]]
        V = [group["id"] for group in nodes["group"]]

        B.add_nodes_from(U, bipartite=0)
        B.add_nodes_from(V, bipartite=1)

        for header in nodes["header"]:
            for i, group in enumerate(nodes["group"]):
                ratio = calc_overlap_ratio(header["box"], group["box"])
                if ratio[0] > 0.0:
                    B.add_edge(header["id"], group["id"], weight=ratio[0])

        pairs = nx.algorithms.matching.max_weight_matching(
            B, maxcardinality=True, weight="weight"
        )

        for a, b in pairs:
            if a in U:
                matched_group_and_header[a] = b
            else:
                matched_group_and_header[b] = a

        matched_group_and_cells = {}
        matched_cell_and_groups = {}
        for group in nodes["group"]:
            for cell in nodes["cell"] + nodes["empty"]:
                if is_contained(
                    group["box"],
                    cell["box"],
                    threshold=0.5,
                ):
                    matched_group_and_cells.setdefault(group["id"], []).append(
                        cell["id"]
                    )

                    if cell["id"] not in matched_cell_and_groups:
                        matched_cell_and_groups[cell["id"]] = []

                    matched_cell_and_groups[cell["id"]].append(group["id"])

        table["matched_header_group"] = matched_group_and_header
        table["matched_group_cells"] = matched_group_and_cells
        table["matched_cell_group"] = matched_cell_and_groups

        print(matched_group_and_header)
        print(matched_cell_and_groups)

    def calc_adjacent_header_dag(self, dag, nodes, group_direction, matching):
        """
        ヘッダーの隣接関係を計算
        グループの方向情報を利用して隣接関係を追加
        例えば、あるグループが水平（H）方向であれば、そのグループに属するヘッダー同士は水平方向の隣接関係を持つ可能性があると判断する
        """

        for node in nodes:
            for potential_parent in nodes:
                if node["id"] == potential_parent["id"]:
                    continue

                potential_parent_group_id = matching.get(potential_parent["id"], None)
                node_group_id = matching.get(node["id"], None)

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

        return dag

    def calc_adjacent_cell_dag(self, dag, nodes, group1, group2):
        """
        セルの隣接関係を計算
        セルが同じサブテーブル or 同じグループに属している場合のみ隣接関係を追加
        """

        for node in nodes:
            for potential_parent in nodes:
                if node["id"] == potential_parent["id"]:
                    continue

                node_group_id = group1.get(node["id"], None)
                potential_parent_id = group1.get(potential_parent["id"], None)

                if node_group_id is None or potential_parent_id is None:
                    continue

                if node_group_id != potential_parent_id:
                    node_group_id = group2.get(node["id"], [])
                    potential_parent_id = group2.get(potential_parent["id"], [])

                    if node_group_id is None or potential_parent_id is None:
                        continue

                    if set(node_group_id) != set(potential_parent_id):
                        continue

                # 左右の隣接判定
                if is_right_adjacent(potential_parent["box"], node["box"]):
                    dag.add_edge(potential_parent["id"], node["id"], dir="R")
                    dag.add_edge(node["id"], potential_parent["id"], dir="L")

                # 上下の隣接判定
                if is_bottom_adjacent(potential_parent["box"], node["box"]):
                    dag.add_edge(potential_parent["id"], node["id"], dir="D")
                    dag.add_edge(node["id"], potential_parent["id"], dir="U")
        return dag

    def convert_to_nodes(self, table):
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

    def calc_adjacent_dag_within_group(self, dag, table, headers, cells):
        """
        グループ内のヘッダーとセルの隣接関係を計算
        """

        cell_groups = table.get("matched_cell_group", {})
        header_groups = table.get("matched_header_group", {})

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
                        group_direction[header_group_id] = "H"

                if is_bottom_adjacent(header["box"], cell["box"]):
                    header_group_id = header_groups.get(header["id"], None)
                    cell_group_ids = cell_groups.get(cell["id"], None)

                    if header_group_id is None or cell_group_ids is None:
                        continue

                    if header_group_id in cell_group_ids:
                        dag.add_edge(header["id"], cell["id"], dir="D")
                        group_direction[header_group_id] = "V"

        return dag, group_direction

    def join_cols_for_sub_table(self, table, nodes, group_direction):
        """
        グループの方向情報を利用して、列をマージし、サブテーブルを生成
        """

        cols = [
            group
            for group in nodes["group"]
            if group["id"] in group_direction and group_direction[group["id"]] == "V"
        ]

        group_cells = table.get("matched_group_cells", {})

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

        print(f"Found {len(sub_tables)} subtables in table.")

        return sub_tables

    def matching_cells_within_sub_tables(self, table, sub_tables):
        matched_cell_and_subtable = {}
        group_cells = table.get("matched_group_cells", {})

        for sub_table in sub_tables:
            member_group_ids = sub_table["member_ids"]

            for group_id in member_group_ids:
                if group_id not in group_cells:
                    continue

                for cell_id in group_cells[group_id]:
                    matched_cell_and_subtable[cell_id] = sub_table["id"]

        table["matched_cell_subtable"] = matched_cell_and_subtable

    def __call__(self, img, id):
        layout_results, vis = self.layout_parser(img)
        table_boxes = [table["box"] for table in layout_results["tables"]]
        table_results, vis_cell, vis_group = self.table_structure_recognizer(
            img, table_boxes, vis=vis
        )

        subtable_img = img.copy()
        for i, table in enumerate(table_results):
            nodes = self.convert_to_nodes(table)
            self.matching_cells(table, nodes)
            debug_matching(i, table, img)

            dag = nx.DiGraph()
            for node in nodes["header"] + nodes["cell"] + nodes["empty"]:
                dag.add_node(node["id"], bbox=node["box"])

            dag, group_direction = self.calc_adjacent_dag_within_group(
                dag, table, nodes["header"], nodes["cell"] + nodes["empty"]
            )

            dag = self.calc_adjacent_header_dag(
                dag,
                nodes["header"],
                group_direction,
                table.get("matched_header_group", {}),
            )

            sub_tables = self.join_cols_for_sub_table(table, nodes, group_direction)

            self.matching_cells_within_sub_tables(table, sub_tables)

            dag = self.calc_adjacent_cell_dag(
                dag,
                nodes["cell"] + nodes["empty"],
                table.get("matched_cell_subtable", {}),
                table.get("matched_cell_group", {}),
            )

            for u, v, attrs in dag.edges(data=True):
                import cv2

                # print(f"{u} -> {v} : {attrs}")

                if attrs["dir"] in ["L", "U"]:
                    continue

                cx1 = (dag.nodes[u]["bbox"][0] + dag.nodes[u]["bbox"][2]) / 2
                cy1 = (dag.nodes[u]["bbox"][1] + dag.nodes[u]["bbox"][3]) / 2
                cx2 = (dag.nodes[v]["bbox"][0] + dag.nodes[v]["bbox"][2]) / 2
                cy2 = (dag.nodes[v]["bbox"][1] + dag.nodes[v]["bbox"][3]) / 2

                color = (0, 255, 0) if attrs["dir"] == "R" else (255, 0, 0)

                vis_cell = cv2.arrowedLine(
                    vis_cell,
                    (int(cx1), int(cy1)),
                    (int(cx2), int(cy2)),
                    color,
                    2,
                )

                cv2.imwrite(f"debug/matched_headers_{id}.png", vis_cell)

            for j, sub_table in enumerate(sub_tables):
                x1, y1, x2, y2 = map(int, sub_table["box"])
                subtable_img = cv2.rectangle(
                    subtable_img,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2,
                )

                cv2.imwrite(f"debug/matched_subtables_{id}.png", subtable_img)

            # print(root_header)

        # results = LayoutAnalyzerSchema(
        #    paragraphs=layout_results.paragraphs,
        #    tables=table_results,
        #    figures=layout_results.figures,
        # )

        return table_results, vis_cell, vis_group


def debug_matching(i, table, img):
    import os

    outputs = "debug"
    os.makedirs(outputs, exist_ok=True)

    matched = table.get("matched_header_group", {})

    for j, (k, v) in enumerate(matched.items()):
        out = img.copy()
        header = next(cell for cell in table["cells"] if cell["id"] == k)
        group = next(cell for cell in table["cells"] if cell["id"] == v)

        cells = table.get("matched_cell_group", {}).get(v, [])

        import cv2

        out = cv2.rectangle(
            out,
            (int(header["box"][0]), int(header["box"][1])),
            (int(header["box"][2]), int(header["box"][3])),
            (0, 255, 255),
            2,
        )

        out = cv2.rectangle(
            out,
            (int(group["box"][0]), int(group["box"][1])),
            (int(group["box"][2]), int(group["box"][3])),
            (255, 0, 255),
            2,
        )

        for cell_id in cells:
            cell = next(cell for cell in table["cells"] if cell["id"] == cell_id)
            out = cv2.rectangle(
                out,
                (int(cell["box"][0]), int(cell["box"][1])),
                (int(cell["box"][2]), int(cell["box"][3])),
                (255, 255, 0),
                2,
            )

        cv2.imwrite(f"{outputs}/matched_{i}_{j}.png", out)
