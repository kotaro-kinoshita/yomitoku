import cv2

import asyncio

import networkx as nx

import numpy as np
from PIL import Image, ImageDraw, ImageFont, features

from concurrent.futures import ThreadPoolExecutor

from .layout_parser import LayoutParser
from .table_cell_detector import CellDetector
from .text_detector import TextDetector
from .text_recognizer import TextRecognizer
from .document_analyzer import extract_words_within_element
from .ocr import OCRSchema, ocr_aggregate

from .utils.visualizer import (
    cell_detector_visualizer,
)
from .utils.misc import (
    is_right_adjacent,
    is_bottom_adjacent,
)

from .grid_parser import parse_grid_from_bottom_up
from .kv_parser import parse_kv_items
from .utils.logger import set_logger


from .schemas.table_semantic_parser import (
    TableSemanticContentsSchema,
    TableSemanticParserSchema,
    CellSchema,
)

from .schemas import TableCellSchema


from .schemas import Element


from typing import Tuple

BBox = Tuple[float, float, float, float]

logger = set_logger(__name__, "INFO")


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


def sort_cells(cells, offset=0):
    if len(cells) == 0:
        return cells, {}

    min_height = min([(cell.box[3] - cell.box[1]) for cell in cells])

    values = [c for c in cells if c.role in ["cell", "header", "empty"]]
    groups = [c for c in cells if c.role == "group"]

    values = sorted(values, key=lambda x: (x.box[1] // min_height, x.box[0]))
    groups = sorted(groups, key=lambda x: (x.box[1], x.box[0]))

    cells = values + groups

    remap_ids = {}
    for i, cell in enumerate(cells):
        new_id = f"c{str(i + offset)}"
        remap_ids[cell.id] = new_id
        cell.id = new_id

    return cells, remap_ids


def _sort_elements(elements, prefix="t"):
    if len(elements) == 0:
        return elements

    min_height = min([(elem.box[3] - elem.box[1]) for elem in elements])

    elements = sorted(elements, key=lambda x: (x.box[1] // min_height, x.box[0]))

    for i, elem in enumerate(elements):
        elem.id = f"{prefix}{str(i)}"
    return elements


def _assign_ids(table_information, cell_offset=0):
    for i, grid in enumerate(table_information["grids"]):
        grid.id = f"g{i}"

    for i, kv in enumerate(table_information["kv_items"]):
        kv.id = f"kv{i}"

    cells, remap_ids = sort_cells(
        table_information["cells"].values(), offset=cell_offset
    )
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

    return cell_offset + len(cells)


def dag_visualizer(dag, img):
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

        self.grid_only = False
        self.merge_same_column_values = False

    def aggregate(self, ocr_res, cells):
        for cell in cells:
            words, direction, _ = extract_words_within_element(ocr_res.words, cell)

            if words is None:
                words = ""

            words = words.replace("\n", "").strip()
            cell.contents = words

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

            for kv_item in results_table.kv_items:
                box = kv_item.box
                cv2.rectangle(
                    vis_layout,
                    (box[0], box[1]),
                    (box[2], box[3]),
                    (0, 0, 255),
                    3,
                )

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

    def __call__(self, img, template=None, id=None):
        results_ocr, results_table, paragraphs = asyncio.run(self.run_models(img))

        semantic_info = []
        for table in results_table:
            self.aggregate(results_ocr, table.cells)

        self.aggregate(results_ocr, paragraphs)

        vis_layout = img.copy()
        vis_ocr = img.copy()

        cell_offset = 0
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
                nodes = _split_nodes_with_role(table.cells)

                if not self.grid_only:
                    clusters, dag = _weakly_cluster_nodes_with_graph(nodes)
                    cluster_nodes_list = _get_cluster_nodes(clusters, nodes)

                else:
                    clusters = [[cell.id for cell in table.cells]]
                    cluster_nodes_list = [nodes]

                for clustered_nodes in cluster_nodes_list:
                    if is_grid_cluster(clustered_nodes):
                        grid, grid_cells, dag = parse_grid_from_bottom_up(
                            cells, clustered_nodes, self.merge_same_column_values
                        )

                        if grid is None:
                            continue

                        table_information["grids"].append(grid)
                        table_information["cells"].update(grid_cells)

                    else:
                        kv_items, dag, kv_cells = parse_kv_items(
                            clustered_nodes,
                            nodes,
                            cells,
                        )

                        table_information["kv_items"].extend(kv_items)
                        table_information["cells"].update(kv_cells)

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

            cell_offset = _assign_ids(table_information, cell_offset)

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

        return semantic_info, vis_layout, vis_ocr
