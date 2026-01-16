import asyncio

import networkx as nx

from concurrent.futures import ThreadPoolExecutor

from .table_detector import TableDetector
from .cell_detector import CellDetector
from .text_detector import TextDetector
from .text_recognizer import TextRecognizer
from .document_analyzer import extract_words_within_element
from .ocr import OCRSchema, ocr_aggregate
from .utils.misc import is_right_adjacent, is_bottom_adjacent


from .utils.visualizer import det_visualizer, dag_visualizer, cell_detector_visualizer
from .utils.misc import (
    replace_spanning_words_with_clipped_polys_poly,
    build_text_detector_schema_from_split_words_rotated_quad,
    box_to_poly,
    quad_to_poly,
)

from .grid_parser import parse_grid_from_bottom_up
from .kv_parser import parse_kv_items

from .constants import PALETTE

from .schemas.table_semantic_parser import (
    TableSemanticContentsSchema,
    TableSemanticParserSchema,
)

from .schemas.document_analyzer import TextDetectorSchema


from typing import Tuple


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


class TableSemanticParser:
    def __init__(
        self,
        configs={},
        device="cuda:1",
        visualize=False,
        dag_visualize=True,
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

    def __call__(self, img, template=None, prediction_kv=True):
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

                if prediction_kv:
                    clusters = _weakly_cluster_nodes_with_graph(nodes)
                    cluster_nodes_list = cluster_nodes(clusters, nodes)
                else:
                    clusters = [[cell.id for cell in table.cells]]
                    cluster_nodes_list = [nodes]

                for cluster, clustered_nodes in zip(clusters, cluster_nodes_list):
                    if not prediction_kv or is_grid_cluster(clustered_nodes):
                        grid, dag = parse_grid_from_bottom_up(
                            table_information["cells"], clustered_nodes
                        )

                        if grid is None:
                            continue

                        table_information["grids"].append(grid)
                    else:
                        kv_items, dag = parse_kv_items(
                            clustered_nodes,
                            nodes,
                        )

                        table_information["kv_items"].extend(kv_items)

                    if self.dag_visualize and self.visualize:
                        vis_cell = dag_visualizer(
                            dag,
                            vis_cell,
                        )

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
