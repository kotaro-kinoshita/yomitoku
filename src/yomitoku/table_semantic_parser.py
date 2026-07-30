import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

import cv2
import networkx as nx
import numpy as np
from PIL import Image, ImageDraw, ImageFont, features

from .grid_parser import parse_grid_from_bottom_up
from .kv_parser import parse_kv_items
from .layout_parser import LayoutParser
from .ocr import OCRSchema, ocr_aggregate
from .reading_order import prediction_reading_order
from .schemas import Element, TableCellSchema
from .schemas.document_analyzer import ParagraphSchema
from .schemas.table_semantic_parser import (
    CellSchema,
    TableSemanticContentsSchema,
    TableSemanticParserSchema,
)
from .table_cell_detector import CellDetector
from .text_detector import TextDetector
from .text_recognizer import TextRecognizer
from .utils.logger import set_logger
from .utils.misc import (
    calc_overlap_ratio,
    is_bottom_adjacent,
    is_contained,
    is_right_adjacent,
    quad_to_xyxy,
)
from .utils.visualizer import cell_detector_visualizer

BBox = Tuple[float, float, float, float]

logger = set_logger(__name__, "INFO")


def _split_nodes_with_role(cells):
    """
    セルの役割ごとにノードを分割
    """

    nodes = {
        "header": [],
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


def _get_cluster_nodes(clusters, nodes):
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


def _weakly_cluster_nodes_with_graph(nodes):
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

    return list(nx.weakly_connected_components(dag)), dag


def _region_cell_ids(region, cells, threshold=0.5):
    """領域 region に内包されるセルの id 集合を返す。"""
    ids = set()
    for cell in cells:
        if is_contained(region.box, cell.box, threshold=threshold):
            ids.add(cell.id)
    return ids


def _resolve_overlapping_regions(
    grid_regions,
    kv_regions,
    value_cells,
    conflict_ratio_th=0.5,
):
    """モデルが kv_item と grid を重複して予測した場合の優先度を解決する。

    領域内のセル網羅度（内包するセル数）を基準に、競合する領域の一方を採用する。
    - 2領域が共有するセルが、いずれかの領域の保持セルの大半 (>= conflict_ratio_th)
      を占める場合に「競合」とみなす。
    - 競合時はセル網羅度が高い方を優先し、同数なら grid を優先する。

    NOTE: 細部のルール（面積比・行列構造の妥当性など）は今後詰める。
    """
    grid_cellsets = [_region_cell_ids(g, value_cells) for g in grid_regions]
    kv_cellsets = [_region_cell_ids(k, value_cells) for k in kv_regions]

    drop_grid = set()
    drop_kv = set()

    # grid同士の重複予測を解決する (同一テーブルへの二重予測)。
    # 共有セルが小さい方の領域の保持セルの大半を占める場合は重複とみなし、
    # 検出スコアが高い方を残す
    score_order = sorted(
        range(len(grid_regions)),
        key=lambda i: getattr(grid_regions[i], "score", 1.0),
        reverse=True,
    )
    kept_grids = []
    for gi in score_order:
        is_duplicate = False
        for gj in kept_grids:
            inter = grid_cellsets[gi] & grid_cellsets[gj]
            denom = min(len(grid_cellsets[gi]), len(grid_cellsets[gj]))
            if denom and len(inter) / denom >= conflict_ratio_th:
                is_duplicate = True
                break
        if is_duplicate:
            drop_grid.add(gi)
        else:
            kept_grids.append(gi)

    for gi in range(len(grid_regions)):
        for ki in range(len(kv_regions)):
            if gi in drop_grid or ki in drop_kv:
                continue

            inter = grid_cellsets[gi] & kv_cellsets[ki]
            if not inter:
                continue

            g_cov = len(grid_cellsets[gi])
            k_cov = len(kv_cellsets[ki])
            denom = min(g_cov, k_cov)
            if denom == 0:
                continue

            if len(inter) / denom < conflict_ratio_th:
                continue

            # セル網羅度が高い方を優先（同数なら grid 優先）
            if g_cov >= k_cov:
                drop_kv.add(ki)
            else:
                drop_grid.add(gi)

    grids = [g for i, g in enumerate(grid_regions) if i not in drop_grid]
    kvs = [k for i, k in enumerate(kv_regions) if i not in drop_kv]
    return grids, kvs


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


def _layout_visualizer(results, img, prefix="Element"):
    vis = img.copy()

    for paragraph in results:
        box = paragraph.box
        id = paragraph.id
        cv2.rectangle(
            vis,
            (box[0], box[1]),
            (box[2], box[3]),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            vis,
            f"{prefix}: {id}",
            (box[0], box[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
        )

    return vis


def _ocr_visualizer(
    img,
    outputs,
    font_path,
    font_size=12,
    font_color=(255, 0, 0),
):
    out = img.copy()
    pillow_img = Image.fromarray(out)
    draw = ImageDraw.Draw(pillow_img)
    has_raqm = features.check_feature(feature="raqm")
    if not has_raqm:
        logger.warning(
            "libraqm is not installed. Vertical text rendering is not supported. Rendering horizontally instead."
        )

    for word in outputs.words:
        quad = np.array(word.points).astype(np.int32)
        font = ImageFont.truetype(font_path, font_size)
        draw.polygon(
            [
                (quad[0][0], quad[0][1]),
                (quad[1][0], quad[1][1]),
                (quad[2][0], quad[2][1]),
                (quad[3][0], quad[3][1]),
            ],
            outline=(0, 255, 0),
            fill=None,
        )

        if word.direction == "horizontal" or not has_raqm:
            x_offset = 0
            y_offset = -font_size

            pos_x = quad[0][0] + x_offset
            pox_y = quad[0][1] + y_offset
            draw.text((pos_x, pox_y), word.content, font=font, fill=font_color)
        else:
            x_offset = -font_size
            y_offset = 0

            pos_x = quad[0][0] + x_offset
            pox_y = quad[0][1] + y_offset
            draw.text(
                (pos_x, pox_y),
                word.content,
                font=font,
                fill=font_color,
                direction="ttb",
            )

    out = np.array(pillow_img)
    return out


def _gap_valley_tol(coords, min_tol=8.0):
    """クラスタ分割の閾値を、隣接ギャップ分布の「谷」から動的に決める。

    同一行 (列) 内の座標ずれは数 px、行 (列) ピッチは数十 px 以上と
    分布が二峰性になるため、隣接ギャップを昇順に並べて最大比ジャンプの
    中点を閾値にする。固定閾値 (セル高の定数倍) だと実ギャップと閾値が
    近接したときに数 px のジッタで分割/結合が反転するが、谷に置くことで
    閾値からの余裕が両側に最大化される。
    """
    coords = sorted(coords)
    gaps = sorted(b - a for a, b in zip(coords, coords[1:]))
    if not gaps or gaps[-1] <= min_tol:
        return min_tol

    # min_tol を基準ギャップとして先頭に加え、単一ギャップでも谷を探せるようにする
    gaps = [min_tol] + gaps
    best_i, best_ratio = 0, 0.0
    for i in range(len(gaps) - 1):
        ratio = (gaps[i + 1] + 1) / (gaps[i] + 1)
        if ratio >= best_ratio:
            best_ratio, best_i = ratio, i
    return (gaps[best_i] + gaps[best_i + 1]) / 2


def _cluster_centers(coords, tol):
    """昇順座標を隣接ギャップ > tol で分割し、クラスタ中心の列を返す"""
    coords = sorted(coords)
    centers = []
    group = [coords[0]]
    for v in coords[1:]:
        if v - group[-1] > tol:
            centers.append(sum(group) / len(group))
            group = [v]
        else:
            group.append(v)
    centers.append(sum(group) / len(group))
    return centers


def _nearest_index(value, centers):
    return min(range(len(centers)), key=lambda i: abs(centers[i] - value))


def sort_cells(cells):
    """セルを読み順に整列し、位置ベースの ID (r{行}c{列}) を割り当てる。

    行インデックスは、テーブル内の非 group セルの上端 y 座標を
    クラスタリングして導出する (grid 由来でない kv セルにも付く)。
    列インデックスは「その行内で左から数えた序数」。x 座標の全体
    クラスタリングにしないのは、行ごとに区切り位置が異なる密な帳票では
    x 分布に谷がなく列クラスタが不安定になるため (行内序数なら検出増減の
    影響はその行の右側のみに閉じる)。連番と違い他の行のセル増減や
    テーブル box に依存せず、座標ジッタ・スキャン位置ずれに対して安定する。
    group ロールのセルは grp0, grp1 の連番。
    """
    if len(cells) == 0:
        return cells, {}

    min_height = min([(cell.box[3] - cell.box[1]) for cell in cells])

    values = [c for c in cells if c.role in ["cell", "header", "empty"]]
    groups = [c for c in cells if c.role == "group"]

    values = sorted(values, key=lambda x: (x.box[1] // min_height, x.box[0]))
    groups = sorted(groups, key=lambda x: (x.box[1], x.box[0]))

    remap_ids = {}

    if values:
        ys = [c.box[1] for c in values]
        row_centers = _cluster_centers(ys, _gap_valley_tol(ys))

        rows = {}
        for cell in values:
            row = _nearest_index(cell.box[1], row_centers)
            rows.setdefault(row, []).append(cell)

        for row, row_cells in rows.items():
            row_cells.sort(key=lambda c: c.box[0])
            for col, cell in enumerate(row_cells):
                new_id = f"r{row}c{col}"
                remap_ids[cell.id] = new_id
                cell.id = new_id

    for i, cell in enumerate(groups):
        new_id = f"grp{i}"
        remap_ids[cell.id] = new_id
        cell.id = new_id

    return values + groups, remap_ids


def _sort_elements(elements, prefix="t"):
    if len(elements) == 0:
        return elements

    min_height = min([(elem.box[3] - elem.box[1]) for elem in elements])

    elements = sorted(elements, key=lambda x: (x.box[1] // min_height, x.box[0]))

    for i, elem in enumerate(elements):
        elem.id = f"{prefix}{str(i)}"
    return elements


def _assign_ids(table_information):
    for i, grid in enumerate(table_information["grids"]):
        grid.id = f"g{i}"

    for i, kv in enumerate(table_information["kv_items"]):
        kv.id = f"kv{i}"

    cells, remap_ids = sort_cells(table_information["cells"].values())
    table_information["cells"] = {cell.id: cell for cell in cells}

    for kv in table_information["kv_items"]:
        new_key = []
        for k in kv.key:
            new_key.append(remap_ids[k])
        kv.key = new_key
        kv.value = remap_ids[kv.value]

    for grid in table_information["grids"]:
        new_grid_data = []
        for row in grid.data:
            new_row = []
            for id in row:
                if id is not None:
                    new_row.append(remap_ids[id])
                else:
                    new_row.append(None)
            new_grid_data.append(new_row)

        new_col_headers = []
        for header in grid.col_headers:
            new_headers = []
            for ck in header:
                if ck is not None:
                    new_headers.append(remap_ids[ck])
                else:
                    new_headers.append(None)
            new_col_headers.append(new_headers)

        grid.data = new_grid_data
        grid.col_headers = new_col_headers


def kv_items_visualizer(table, img):
    """確定した kv_items のキー連鎖 (key[0] -> ... -> value) を緑の矢印で描画する。

    DAGではなく最終的な kv_items から描画するため、フォールバック救済で
    連結されたキー (孤児ヘッダー・孤児セル連鎖) も含めて、確定した
    KV関係がすべて可視化される。
    """
    for kv in table.kv_items:
        keys = [kv.key] if isinstance(kv.key, str) else list(kv.key)
        chain = [k for k in keys if k in table.cells]
        if kv.value in table.cells:
            chain.append(kv.value)

        for u, v in zip(chain, chain[1:]):
            bu = table.cells[u].box
            bv = table.cells[v].box
            cx1, cy1 = (bu[0] + bu[2]) / 2, (bu[1] + bu[3]) / 2
            cx2, cy2 = (bv[0] + bv[2]) / 2, (bv[1] + bv[3]) / 2

            # dag_visualizer と同じ規則: 隣接方向と直交する軸は両セルの
            # 重なり帯の中点に揃えて水平/垂直に描画する
            y_lo, y_hi = max(bu[1], bv[1]), min(bu[3], bv[3])
            x_lo, x_hi = max(bu[0], bv[0]), min(bu[2], bv[2])
            if y_lo < y_hi and int(cx1) != int(cx2):
                cy1 = cy2 = (y_lo + y_hi) / 2
            elif x_lo < x_hi:
                cx1 = cx2 = (x_lo + x_hi) / 2

            length = max(1.0, ((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2) ** 0.5)
            tip_length = min(0.2, 12.0 / length)

            img = cv2.arrowedLine(
                img,
                (int(cx1), int(cy1)),
                (int(cx2), int(cy2)),
                (0, 255, 0),
                2,
                tipLength=tip_length,
            )

    return img


def dag_visualizer(dag, img):
    for u, v, attrs in dag.edges(data=True):
        if attrs["dir"] in ["L", "U"]:
            continue
        bu = dag.nodes[u]["bbox"]
        bv = dag.nodes[v]["bbox"]
        cx1 = (bu[0] + bu[2]) / 2
        cy1 = (bu[1] + bu[3]) / 2
        cx2 = (bv[0] + bv[2]) / 2
        cy2 = (bv[1] + bv[3]) / 2

        # 片方がスパンセルだと中心同士を結ぶ線は斜めになり実際の接続位置と
        # ずれるため、隣接方向と直交する軸は「両セルの重なり帯の中点」に
        # 揃えて水平/垂直に描画する（重なりが無い異常エッジは中心のまま
        # 斜めに描画され、目視で検出できる）
        if attrs["dir"] == "R":
            lo = max(bu[1], bv[1])
            hi = min(bu[3], bv[3])
            if lo < hi:
                cy1 = cy2 = (lo + hi) / 2
        elif attrs["dir"] == "D":
            lo = max(bu[0], bv[0])
            hi = min(bu[2], bv[2])
            if lo < hi:
                cx1 = cx2 = (lo + hi) / 2

        # 矢印チップは線長比だと長いエッジで巨大化するため、絶対長で揃える
        length = max(1.0, ((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2) ** 0.5)
        tip_length = min(0.2, 12.0 / length)

        color = (0, 255, 0) if attrs["dir"] == "R" else (255, 0, 0)
        img = cv2.arrowedLine(
            img,
            (int(cx1), int(cy1)),
            (int(cx2), int(cy2)),
            color,
            2,
            tipLength=tip_length,
        )

    return img


class TableSemanticParser:
    def __init__(
        self,
        configs={},
        device="cuda:0",
        visualize=True,
    ):
        table_detector_kwargs = {
            "device": device,
            "visualize": visualize,
        }
        table_cell_parser_kwargs = {
            "device": device,
            "visualize": visualize,
        }

        text_detector_kwargs = {
            "device": device,
        }

        text_recognizer_kwargs = {
            "device": device,
        }

        if isinstance(configs, dict):
            if "table_detector" in configs:
                table_detector_kwargs.update(configs["table_detector"])

            if "table_cell_parser" in configs:
                table_cell_parser_kwargs.update(configs["table_cell_parser"])

            if "text_detector" in configs:
                text_detector_kwargs.update(configs["text_detector"])

            if "text_recognizer" in configs:
                text_recognizer_kwargs.update(configs["text_recognizer"])
        else:
            raise ValueError(
                "configs must be a dict. See the https://kotaro-kinoshita.github.io/yomitoku/module/#config"
            )

        self.layout_parser = LayoutParser(
            **table_detector_kwargs,
        )
        self.cell_detector = CellDetector(
            **table_cell_parser_kwargs,
        )

        self.text_detector = TextDetector(
            **text_detector_kwargs,
        )

        self.text_recognizer = TextRecognizer(
            **text_recognizer_kwargs,
        )

        self.visualize = visualize

        self.merge_same_column_values = False

    def aggregate(self, ocr_res, cells, overlap_th=0.2):
        from collections import defaultdict

        cell_words = defaultdict(list)

        for word in ocr_res.words:
            word_box = quad_to_xyxy(word.points)
            best_cell = None
            best_ratio = 0

            for cell in cells:
                if cell.role == "group":
                    continue
                ratio, _ = calc_overlap_ratio(cell.box, word_box)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_cell = cell

            if best_cell is None or best_ratio < overlap_th:
                continue

            word_element = ParagraphSchema(
                box=word_box,
                contents=word.content,
                direction=word.direction,
                order=0,
                role=None,
            )
            # 段落要素は id が未採番(None)のまま渡されるため、cell.id を
            # キーにすると全要素が同一キーに集約され、全単語が全段落に
            # 割り当たってしまう。オブジェクト同一性をキーにする。
            cell_words[id(best_cell)].append(word_element)

        for cell in cells:
            contained = cell_words.get(id(cell), [])
            if not contained:
                cell.contents = ""
                continue

            dirs = [w.direction for w in contained]
            direction = (
                "horizontal"
                if dirs.count("horizontal") >= dirs.count("vertical")
                else "vertical"
            )
            order = "left2right" if direction == "horizontal" else "right2left"
            prediction_reading_order(contained, order)
            contained = sorted(contained, key=lambda x: x.order)
            text = "\n".join([w.contents for w in contained])
            cell.contents = text.replace("\n", "").strip()

    def replace_table_to_paragraphs(self, tables, paragraphs):
        new_table_list = []
        for table in tables:
            cnt_cell = 0
            for cell in table.cells:
                if cell.role in ["cell", "header"]:
                    cnt_cell += 1

            if cnt_cell < 2:
                paragraphs.append(
                    Element(
                        id=None,
                        box=table.box,
                        contents="",
                        score=1.0,
                        role=None,
                    )
                )
            else:
                new_table_list.append(table)

        return new_table_list

    async def run_models(self, img):
        with ThreadPoolExecutor(max_workers=2) as executor:
            loop = asyncio.get_running_loop()
            tasks = [
                loop.run_in_executor(executor, self.text_detector, img),
                loop.run_in_executor(executor, self.layout_parser, img),
            ]

            results = await asyncio.gather(*tasks)

        results_det, _ = results[0]
        results_layout, _ = results[1]

        bordered_table = [t for t in results_layout.tables]

        results_table = self.cell_detector(img, bordered_table)

        results_table = self.replace_table_to_paragraphs(
            results_table, results_layout.paragraphs
        )

        results_rec, _ = self.text_recognizer(img, results_det.points)
        outputs = {"words": ocr_aggregate(results_det, results_rec)}
        results_ocr = OCRSchema(**outputs)

        return (results_ocr, results_table, results_layout.paragraphs)

    def visualizer_ocr(self, img, semantic_info):
        vis_ocr = _ocr_visualizer(
            img,
            semantic_info,
            font_size=self.text_recognizer._cfg.visualize.font_size,
            font_color=tuple(self.text_recognizer._cfg.visualize.color[::-1]),
            font_path=self.text_recognizer._cfg.visualize.font,
        )

        return vis_ocr

    def visualizer_layout(self, img, semantic_info):
        vis_layout = img.copy()

        vis_layout = _layout_visualizer(
            semantic_info.tables,
            vis_layout,
            prefix="Table",
        )

        vis_layout = _layout_visualizer(
            semantic_info.paragraphs,
            vis_layout,
            prefix="Paragraph",
        )

        for results_table in semantic_info.tables:
            vis_layout, _ = cell_detector_visualizer(
                vis_layout,
                vis_layout,
                results_table.cells.values(),
            )

            # kv_items のキー連鎖を緑の矢印で描画する (救済リンク含む)
            vis_layout = kv_items_visualizer(results_table, vis_layout)

            for grid in results_table.grids:
                box = grid.box
                cv2.rectangle(
                    vis_layout,
                    (box[0], box[1]),
                    (box[2], box[3]),
                    (255, 0, 0),
                    3,
                )

        return vis_layout

    def __call__(self, img, template=None, id=None, grid_only=False, kv_only=False):
        results_ocr, results_table, paragraphs = asyncio.run(self.run_models(img))

        semantic_info = []
        for table in results_table:
            self.aggregate(results_ocr, table.cells)

        self.aggregate(results_ocr, paragraphs)

        vis_layout = img.copy()
        vis_ocr = img.copy()

        # テーブル構造のグラフ (DAG) を最後にまとめて上描きするため保持する
        dags = []

        for i, table in enumerate(results_table):
            cells = {}
            for cell in table.cells:
                if isinstance(cell, TableCellSchema):
                    cell = CellSchema(
                        meta={},
                        id=cell.id,
                        box=cell.box,
                        role=cell.role,
                        row=cell.row,
                        col=cell.col,
                        row_span=cell.row_span,
                        col_span=cell.col_span,
                        contents=cell.contents,
                    )

                cells[cell.id] = cell

            table_information = {
                "id": f"t{i}",
                "box": table.box,
                "cells": {},
                "style": "border",
                "kv_items": [],
                "grids": [],
            }
            if template is None:
                # モデルが直接予測した grid / kv_item 領域を利用する
                value_cells = [
                    c for c in table.cells if c.role in ("cell", "header", "empty")
                ]

                grid_regions = list(table.grid_regions)
                kv_regions = list(table.kv_regions)

                if grid_only:
                    kv_regions = []
                if kv_only:
                    grid_regions = []

                # kv_item と grid が重複検知された場合の優先度をルールベースで解決
                grid_regions, kv_regions = _resolve_overlapping_regions(
                    grid_regions,
                    kv_regions,
                    value_cells,
                )

                # --- grid 領域ごとに内部構造（行列）を推定 ---
                grid_claimed_ids = set()
                for region in grid_regions:
                    region_cells = [
                        c
                        for c in value_cells
                        if is_contained(region.box, c.box, threshold=0.5)
                    ]
                    if len(region_cells) == 0:
                        continue

                    clustered_nodes = _split_nodes_with_role(region_cells)
                    result = parse_grid_from_bottom_up(
                        cells,
                        clustered_nodes,
                        self.merge_same_column_values,
                    )

                    if result is None:
                        continue

                    grid, grid_cells, dag = result

                    table_information["grids"].append(grid)
                    table_information["cells"].update(grid_cells)
                    grid_claimed_ids.update(c.id for c in region_cells)

                    dags.append(dag)

                # --- grid に属さないセルを kv として推定 ---
                remaining_cells = [
                    c for c in value_cells if c.id not in grid_claimed_ids
                ]
                if remaining_cells:
                    nodes = _split_nodes_with_role(remaining_cells)

                    # モデル予測の kv_item 領域内のセルで隣接グラフを構築する
                    for ki, region in enumerate(kv_regions):
                        region.id = f"kvr{ki}"

                    kv_items, dag, kv_cells = parse_kv_items(
                        nodes,
                        cells,
                        kv_regions,
                    )

                    table_information["kv_items"].extend(kv_items)
                    table_information["cells"].update(kv_cells)

                    # NOTE: kv側はDAGではなく確定した kv_items のキー連鎖を
                    # kv_items_visualizer (緑矢印) で描画するため、DAGの
                    # 上描き対象には追加しない (dagsはgrid構造の可視化用)

            for cell in cells.values():
                if cell.id not in table_information["cells"]:
                    table_information["cells"][cell.id] = cell

            table_information["kv_items"] = sorted(
                table_information["kv_items"],
                key=lambda kv: table_information["cells"][kv.value].box[1],
            )

            table_information["grids"] = sorted(
                table_information["grids"],
                key=lambda g: g.box[1],
            )

            for i, grid in enumerate(table_information["grids"]):
                grid.id = f"g{i}"

            for i, kv in enumerate(table_information["kv_items"]):
                kv.id = f"kv{i}"

            _assign_ids(table_information)

            semantic_info.append(TableSemanticContentsSchema(**table_information))

        semantic_info = _sort_elements(semantic_info, prefix="t")
        paragraphs = _sort_elements(paragraphs, prefix="p")

        semantic_info = TableSemanticParserSchema(
            tables=semantic_info,
            paragraphs=paragraphs,
            words=results_ocr.words,
        )

        if template is not None:
            semantic_info.load_template_json(template)

        if self.visualize:
            vis_layout = self.visualizer_layout(vis_layout, semantic_info)
            vis_ocr = self.visualizer_ocr(vis_ocr, semantic_info)

            # テーブル構造のグラフ (DAG) をセル塗り・枠の上に描画して見やすくする
            for dag in dags:
                vis_layout = dag_visualizer(dag, vis_layout)

        return semantic_info, vis_layout, vis_ocr
