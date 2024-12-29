import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union

import numpy as np

from pydantic import conlist

from .base import BaseSchema
from .export import export_csv, export_html, export_markdown
from .layout_analyzer import LayoutAnalyzer
from .ocr import OCRSchema, WordPrediction, ocr_aggregate
from .reading_order import prediction_reading_order
from .table_structure_recognizer import TableStructureRecognizerSchema
from .utils.misc import (
    is_contained,
    quad_to_xyxy,
    calc_overlap_ratio,
)
from .utils.visualizer import reading_order_visualizer
from yomitoku.text_detector import TextDetector
from yomitoku.text_recognizer import TextRecognizer

from .utils.visualizer import det_visualizer


class ParagraphSchema(BaseSchema):
    box: conlist(int, min_length=4, max_length=4)
    contents: Union[str, None]
    direction: Union[str, None]
    order: Union[int, None]
    role: Union[str, None]


class FigureSchema(BaseSchema):
    box: conlist(int, min_length=4, max_length=4)
    order: Union[int, None]
    paragraphs: List[ParagraphSchema]
    order: Union[int, None]
    direction: Union[str, None]


class DocumentAnalyzerSchema(BaseSchema):
    paragraphs: List[ParagraphSchema]
    tables: List[TableStructureRecognizerSchema]
    words: List[WordPrediction]
    figures: List[FigureSchema]

    def to_html(self, out_path: str, **kwargs):
        export_html(self, out_path, **kwargs)

    def to_markdown(self, out_path: str, **kwargs):
        export_markdown(self, out_path, **kwargs)

    def to_csv(self, out_path: str, **kwargs):
        export_csv(self, out_path, **kwargs)


def combine_flags(flag1, flag2):
    return [f1 or f2 for f1, f2 in zip(flag1, flag2)]


def judge_page_direction(paragraphs):
    h_sum_area = 0
    v_sum_area = 0

    for paragraph in paragraphs:
        x1, y1, x2, y2 = paragraph.box
        w = x2 - x1
        h = y2 - y1

        if paragraph.direction == "horizontal":
            h_sum_area += w * h
        else:
            v_sum_area += w * h

    if v_sum_area > h_sum_area:
        return "vertical"

    return "horizontal"


def extract_paragraph_within_figure(paragraphs, figures):
    new_figures = []
    check_list = [False] * len(paragraphs)
    for figure in figures:
        figure = {"box": figure.box, "order": 0}
        contained_paragraphs = []
        for i, paragraph in enumerate(paragraphs):
            if is_contained(figure["box"], paragraph.box, threshold=0.7):
                contained_paragraphs.append(paragraph)
                check_list[i] = True

        figure["direction"] = judge_page_direction(contained_paragraphs)
        figure_paragraphs = prediction_reading_order(
            contained_paragraphs, figure["direction"]
        )
        figure["paragraphs"] = sorted(figure_paragraphs, key=lambda x: x.order)
        figure = FigureSchema(**figure)
        new_figures.append(figure)

    return new_figures, check_list


def extract_words_within_element(pred_words, element):
    contained_words = []
    word_sum_width = 0
    word_sum_height = 0
    check_list = [False] * len(pred_words)
    for i, word in enumerate(pred_words):
        word_box = quad_to_xyxy(word.points)
        if is_contained(element.box, word_box, threshold=0.5):
            contained_words.append(word)
            word_sum_width += word_box[2] - word_box[0]
            word_sum_height += word_box[3] - word_box[1]
            check_list[i] = True

    if len(contained_words) == 0:
        return None, None, check_list

    # mean_width = word_sum_width / len(contained_words)
    # mean_height = word_sum_height / len(contained_words)

    element_direction = "horizontal"
    word_direction = [word.direction for word in contained_words]
    cnt_horizontal = word_direction.count("horizontal")
    cnt_vertical = word_direction.count("vertical")

    element_direction = "horizontal" if cnt_horizontal > cnt_vertical else "vertical"
    if element_direction == "horizontal":
        contained_words = sorted(
            contained_words,
            key=lambda x: (sum([p[1] for p in x.points]) / 4),
        )
    else:
        contained_words = sorted(
            contained_words,
            key=lambda x: (sum([p[0] for p in x.points]) / 4),
            reverse=True,
        )

    contained_words = "\n".join([content.content for content in contained_words])

    return (contained_words, element_direction, check_list)


def is_vertical(quad, thresh_aspect=2):
    quad = np.array(quad)
    width = np.linalg.norm(quad[0] - quad[1])
    height = np.linalg.norm(quad[1] - quad[2])

    return height > width * thresh_aspect


def is_noise(quad, thresh=15):
    quad = np.array(quad)
    width = np.linalg.norm(quad[0] - quad[1])
    height = np.linalg.norm(quad[1] - quad[2])

    return width < thresh or height < thresh


def recursive_update(original, new_data):
    for key, value in new_data.items():
        # `value`が辞書の場合、再帰的に更新
        if (
            isinstance(value, dict)
            and key in original
            and isinstance(original[key], dict)
        ):
            recursive_update(original[key], value)
        # `value`が辞書でない場合、またはキーが存在しない場合に上書き
        else:
            original[key] = value
    return original


def _extract_words_within_table(words, table, check_list):
    horizontal_words = []
    vertical_words = []

    for i, (points, score) in enumerate(zip(words.points, words.scores)):
        word_box = quad_to_xyxy(points)
        if is_contained(table.box, word_box, threshold=0.5):
            if is_vertical(points):
                vertical_words.append({"points": points, "score": score})
            else:
                horizontal_words.append({"points": points, "score": score})

            check_list[i] = True

    return (horizontal_words, vertical_words, check_list)


def _calc_overlap_words_on_lines(lines, words):
    overlap_ratios = [[0 for _ in lines] for _ in words]

    for i, word in enumerate(words):
        word_box = quad_to_xyxy(word["points"])
        for j, row in enumerate(lines):
            overlap_ratio, _ = calc_overlap_ratio(
                row.box,
                word_box,
            )
            overlap_ratios[i][j] = overlap_ratio

    return overlap_ratios


def _correct_vertical_word_boxes(overlap_ratios_vertical, table, table_words_vertical):
    allocated_cols = [cols.index(max(cols)) for cols in overlap_ratios_vertical]

    new_points = []
    new_scores = []
    for i, col_index in enumerate(allocated_cols):
        col_cells = []
        for cell in table.cells:
            if cell.col <= (col_index + 1) < (cell.col + cell.col_span):
                col_cells.append(cell)

        word_point = table_words_vertical[i]["points"]
        word_score = table_words_vertical[i]["score"]

        for cell in col_cells:
            word_box = quad_to_xyxy(word_point)

            _, intersection = calc_overlap_ratio(
                cell.box,
                word_box,
            )

            if intersection is not None:
                _, y1, _, y2 = intersection

                new_point = [
                    [word_point[0][0], max(word_point[0][1], y1)],
                    [word_point[1][0], max(word_point[1][1], y1)],
                    [word_point[2][0], min(word_point[2][1], y2)],
                    [word_point[3][0], min(word_point[3][1], y2)],
                ]

                if not is_noise(new_point):
                    new_points.append(new_point)
                    new_scores.append(word_score)

    return new_points, new_scores


def _correct_horizontal_word_boxes(
    overlap_ratios_horizontal, table, table_words_horizontal
):
    allocated_rows = [rows.index(max(rows)) for rows in overlap_ratios_horizontal]

    new_points = []
    new_scores = []
    for i, row_index in enumerate(allocated_rows):
        row_cells = []
        for cell in table.cells:
            if cell.row <= (row_index + 1) < (cell.row + cell.row_span):
                row_cells.append(cell)

        word_point = table_words_horizontal[i]["points"]
        word_score = table_words_horizontal[i]["score"]

        for cell in row_cells:
            word_box = quad_to_xyxy(word_point)

            _, intersection = calc_overlap_ratio(
                cell.box,
                word_box,
            )

            if intersection is not None:
                x1, _, x2, _ = intersection

                new_point = [
                    [max(word_point[0][0], x1), word_point[0][1]],
                    [min(word_point[1][0], x2), word_point[1][1]],
                    [min(word_point[2][0], x2), word_point[2][1]],
                    [max(word_point[3][0], x1), word_point[3][1]],
                ]

                if not is_noise(new_point):
                    new_points.append(new_point)
                    new_scores.append(word_score)

    return new_points, new_scores


def _split_text_across_cells(results_det, results_layout):
    check_list = [False] * len(results_det.points)
    new_points = []
    new_scores = []
    for table in results_layout.tables:
        table_words_horizontal, table_words_vertical, check_list = (
            _extract_words_within_table(results_det, table, check_list)
        )

        overlap_ratios_horizontal = _calc_overlap_words_on_lines(
            table.rows,
            table_words_horizontal,
        )

        overlap_ratios_vertical = _calc_overlap_words_on_lines(
            table.cols,
            table_words_vertical,
        )

        new_points_horizontal, new_scores_horizontal = _correct_horizontal_word_boxes(
            overlap_ratios_horizontal, table, table_words_horizontal
        )

        new_points_vertical, new_scores_vertical = _correct_vertical_word_boxes(
            overlap_ratios_vertical, table, table_words_vertical
        )

        new_points.extend(new_points_horizontal)
        new_scores.extend(new_scores_horizontal)
        new_points.extend(new_points_vertical)
        new_scores.extend(new_scores_vertical)

    for i, flag in enumerate(check_list):
        if not flag:
            new_points.append(results_det.points[i])
            new_scores.append(results_det.scores[i])

    results_det.points = new_points
    results_det.scores = new_scores

    return results_det


class DocumentAnalyzer:
    def __init__(self, configs={}, device="cuda", visualize=False):
        default_configs = {
            "ocr": {
                "text_detector": {
                    "device": device,
                    "visualize": visualize,
                },
                "text_recognizer": {
                    "device": device,
                    "visualize": visualize,
                },
            },
            "layout_analyzer": {
                "layout_parser": {
                    "device": device,
                    "visualize": visualize,
                },
                "table_structure_recognizer": {
                    "device": device,
                    "visualize": visualize,
                },
            },
        }

        if isinstance(configs, dict):
            recursive_update(default_configs, configs)
        else:
            raise ValueError(
                "configs must be a dict. See the https://kotaro-kinoshita.github.io/yomitoku-dev/usage/"
            )

        self.text_detector = TextDetector(
            **default_configs["ocr"]["text_detector"],
        )
        self.text_recognizer = TextRecognizer(
            **default_configs["ocr"]["text_recognizer"]
        )

        self.layout = LayoutAnalyzer(
            configs=default_configs["layout_analyzer"],
        )
        self.visualize = visualize

    def aggregate(self, ocr_res, layout_res):
        paragraphs = []
        check_list = [False] * len(ocr_res.words)
        for table in layout_res.tables:
            for cell in table.cells:
                words, direction, flags = extract_words_within_element(
                    ocr_res.words, cell
                )

                if words is None:
                    words = ""

                cell.contents = words
                check_list = combine_flags(check_list, flags)

        for paragraph in layout_res.paragraphs:
            words, direction, flags = extract_words_within_element(
                ocr_res.words, paragraph
            )

            if words is None:
                continue

            paragraph = {
                "contents": words,
                "box": paragraph.box,
                "direction": direction,
                "order": 0,
                "role": paragraph.role,
            }

            check_list = combine_flags(check_list, flags)
            paragraph = ParagraphSchema(**paragraph)
            paragraphs.append(paragraph)

        for i, word in enumerate(ocr_res.words):
            direction = word.direction
            if not check_list[i]:
                paragraph = {
                    "contents": word.content,
                    "box": quad_to_xyxy(word.points),
                    "direction": direction,
                    "order": 0,
                    "role": None,
                }

                paragraph = ParagraphSchema(**paragraph)
                paragraphs.append(paragraph)

        figures, check_list = extract_paragraph_within_figure(
            paragraphs, layout_res.figures
        )

        paragraphs = [
            paragraph for paragraph, flag in zip(paragraphs, check_list) if not flag
        ]

        page_direction = judge_page_direction(paragraphs)

        headers = [
            paragraph for paragraph in paragraphs if paragraph.role == "page_header"
        ]

        footers = [
            paragraph for paragraph in paragraphs if paragraph.role == "page_footer"
        ]

        page_contents = [
            paragraph
            for paragraph in paragraphs
            if paragraph.role is None or paragraph.role == "section_headings"
        ]

        elements = page_contents + layout_res.tables + figures

        prediction_reading_order(headers, page_direction)
        prediction_reading_order(footers, page_direction)
        prediction_reading_order(elements, page_direction, self.img)

        for i, element in enumerate(elements):
            element.order += len(headers)
        for i, footer in enumerate(footers):
            footer.order += len(elements) + len(headers)

        paragraphs = headers + page_contents + footers
        paragraphs = sorted(paragraphs, key=lambda x: x.order)
        figures = sorted(figures, key=lambda x: x.order)
        tables = sorted(layout_res.tables, key=lambda x: x.order)

        outputs = {
            "paragraphs": paragraphs,
            "tables": tables,
            "figures": figures,
            "words": ocr_res.words,
        }

        return outputs

    async def run(self, img):
        with ThreadPoolExecutor(max_workers=2) as executor:
            loop = asyncio.get_running_loop()
            tasks = [
                # loop.run_in_executor(executor, self.ocr, img),
                loop.run_in_executor(executor, self.text_detector, img),
                loop.run_in_executor(executor, self.layout, img),
            ]

            results = await asyncio.gather(*tasks)

            results_det, _ = results[0]
            results_layout, layout = results[1]

            results_det = _split_text_across_cells(results_det, results_layout)

            vis_det = None
            if self.visualize:
                vis_det = det_visualizer(
                    img,
                    results_det.points,
                )

            results_rec, ocr = self.text_recognizer(img, results_det.points, vis_det)

            outputs = {"words": ocr_aggregate(results_det, results_rec)}
            results_ocr = OCRSchema(**outputs)
            outputs = self.aggregate(results_ocr, results_layout)

        results = DocumentAnalyzerSchema(**outputs)
        return results, ocr, layout

    def __call__(self, img):
        self.img = img
        resutls, ocr, layout = asyncio.run(self.run(img))

        if self.visualize:
            layout = reading_order_visualizer(layout, resutls)

        return resutls, ocr, layout
