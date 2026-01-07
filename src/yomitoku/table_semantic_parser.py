import asyncio
from concurrent.futures import ThreadPoolExecutor

from .table_detector import TableDetector
from .cell_detector import CellDetector
from .text_detector import TextDetector
from .text_recognizer import TextRecognizer
from .document_analyzer import extract_words_within_element
from .ocr import OCRSchema, ocr_aggregate
from .cell_relation import calc_cell_relation_dag
from .parse_table_semantic_infomation import parse_semantic_table_information

from .utils.visualizer import det_visualizer, dag_visualizer
from .utils.misc import (
    replace_spanning_words_with_clipped_polys,
    build_text_detector_schema_from_split_words,
)

from .schemas.table_semantic_parser import (
    TableSemanticContentsSchema,
    TableSemanticParserSchema,
)

from .schemas.document_analyzer import TextDetectorSchema


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


class TableSemanticParser:
    def __init__(
        self, configs={}, device="cuda:1", visualize=False, dag_visualize=False
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

        # セルにまたがるテキスト領域の分割
        split_words = replace_spanning_words_with_clipped_polys(
            words=results_det.points,
            cells=[{"id": t.id, "box": t.box} for t in results_table],
            min_area_ratio=0.03,
            keep_unsplit=True,
        )

        schema_dict = build_text_detector_schema_from_split_words(split_words)
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

        return results_ocr, results_table, vis_cell, vis_group, vis_ocr

    def __call__(self, img):
        results_ocr, results_table, vis_cell, vis_group, vis_ocr = asyncio.run(
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

            nodes = _split_nodes_with_role(table.cells)

            # セル間の関係をDAGで表現
            cell_relation_dag, match, group_direction, grid_regions = (
                calc_cell_relation_dag(nodes)
            )

            # 解析してKVアイテムとグリッドを抽出
            grids, kv_items = parse_semantic_table_information(
                cell_relation_dag,
                match,
                nodes,
                grid_regions,
                group_direction,
            )

            table_information["kv_items"].extend(kv_items)
            table_information["grids"].extend(grids)
            semantic_info.append(TableSemanticContentsSchema(**table_information))

            # For Debug Visualization
            if self.dag_visualize:
                vis_cell = dag_visualizer(cell_relation_dag, vis_cell)

        semantic_info = TableSemanticParserSchema(
            tables=semantic_info,
            words=results_ocr.words,
        )
        return semantic_info, vis_cell, vis_group, vis_ocr
