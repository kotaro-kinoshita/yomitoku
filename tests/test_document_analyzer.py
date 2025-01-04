import pytest
import torch
from omegaconf import OmegaConf

from yomitoku import DocumentAnalyzer
from yomitoku.document_analyzer import (
    ParagraphSchema,
    FigureSchema,
    DocumentAnalyzerSchema,
    extract_paragraph_within_figure,
    combine_flags,
    judge_page_direction,
    extract_words_within_element,
    is_vertical,
    is_noise,
    recursive_update,
    _extract_words_within_table,
    _calc_overlap_words_on_lines,
    _correct_vertical_word_boxes,
    _correct_horizontal_word_boxes,
    _split_text_across_cells,
)


from yomitoku.text_detector import TextDetectorSchema

from yomitoku.table_structure_recognizer import (
    TableStructureRecognizerSchema,
    TableLineSchema,
    TableCellSchema,
)

from yomitoku.ocr import (
    WordPrediction,
)


def test_initialize():
    device = "cpu"
    visualize = True
    config = {
        "ocr": {
            "text_detector": {
                "path_cfg": "tests/yaml/text_detector.yaml",
            },
            "text_recognizer": {
                "path_cfg": "tests/yaml/text_recognizer.yaml",
            },
        },
        "layout_analyzer": {
            "layout_parser": {
                "path_cfg": "tests/yaml/layout_parser.yaml",
            },
            "table_structure_recognizer": {
                "path_cfg": "tests/yaml/table_structure_recognizer.yaml",
            },
        },
    }

    analyzer = DocumentAnalyzer(configs=config, device=device, visualize=visualize)

    # サブモジュールのパラメータが更新されているか確認
    assert analyzer.text_detector.device == torch.device(device)
    assert analyzer.text_recognizer.device == torch.device(device)
    assert analyzer.layout.layout_parser.device == torch.device(device)
    assert analyzer.layout.table_structure_recognizer.device == torch.device(device)

    assert analyzer.text_detector.visualize == visualize
    assert analyzer.text_recognizer.visualize == visualize
    assert analyzer.layout.layout_parser.visualize == visualize
    assert analyzer.layout.table_structure_recognizer.visualize == visualize

    text_detector_cfg = OmegaConf.load(config["ocr"]["text_detector"]["path_cfg"])
    text_recognizer_cfg = OmegaConf.load(config["ocr"]["text_recognizer"]["path_cfg"])
    layout_parser_cfg = OmegaConf.load(
        config["layout_analyzer"]["layout_parser"]["path_cfg"]
    )
    table_structure_recognizer_cfg = OmegaConf.load(
        config["layout_analyzer"]["table_structure_recognizer"]["path_cfg"]
    )

    assert (
        analyzer.text_detector.post_processor.thresh
        == text_detector_cfg.post_process.thresh
    )

    assert (
        analyzer.text_recognizer.model.refine_iters == text_recognizer_cfg.refine_iters
    )

    assert analyzer.layout.layout_parser.thresh_score == layout_parser_cfg.thresh_score

    assert (
        analyzer.layout.table_structure_recognizer.thresh_score
        == table_structure_recognizer_cfg.thresh_score
    )


def test_invalid_path():
    config = {
        "ocr": {
            "text_detector": {
                "path_cfg": "tests/yaml/dummy.yaml",
            },
        }
    }

    with pytest.raises(FileNotFoundError):
        DocumentAnalyzer(
            configs=config,
        )


def test_invalid_config():
    with pytest.raises(ValueError):
        DocumentAnalyzer(
            configs="invalid",
        )


def test_extract_paragraph_within_figure():
    paragraphs = [
        {
            "box": [0, 0, 2, 1],
            "contents": "This is a test.",
            "direction": "horizontal",
            "order": 1,
            "role": None,
        },
        {
            "box": [0, 0, 1, 2],
            "contents": "This is a test.",
            "direction": "vertical",
            "order": 1,
            "role": None,
        },
        {
            "box": [10, 10, 1, 2],
            "contents": "This is a test.",
            "direction": "horizontal",
            "order": 1,
            "role": None,
        },
    ]

    figures = [
        {
            "box": [0, 0, 2, 2],
            "order": 1,
            "paragraphs": [],
            "direction": None,
        }
    ]

    paragraphs = [ParagraphSchema(**paragraph) for paragraph in paragraphs]
    figures = [FigureSchema(**figure) for figure in figures]

    figures, checklist = extract_paragraph_within_figure(paragraphs, figures)

    assert checklist == [True, True, False]
    assert len(figures[0].paragraphs) == 2


def test_combile_flags():
    flags1 = [True, False, True]
    flags2 = [False, False, True]

    assert combine_flags(flags1, flags2) == [True, False, True]


def test_judge_page_direction():
    paragraphs = [
        {
            "box": [0, 0, 2, 1],
            "contents": "This is a test.",
            "direction": "horizontal",
            "order": 1,
            "role": None,
        },
        {
            "box": [0, 0, 1, 2],
            "contents": "This is a test.",
            "direction": "vertical",
            "order": 1,
            "role": None,
        },
        {
            "box": [10, 10, 1, 2],
            "contents": "This is a test.",
            "direction": "horizontal",
            "order": 1,
            "role": None,
        },
    ]

    paragraphs = [ParagraphSchema(**paragraph) for paragraph in paragraphs]
    assert judge_page_direction(paragraphs) == "horizontal"

    paragraphs = [
        {
            "box": [0, 0, 2, 1],
            "contents": "This is a test.",
            "direction": "horizontal",
            "order": 1,
            "role": None,
        },
        {
            "box": [0, 0, 1, 2],
            "contents": "This is a test.",
            "direction": "vertical",
            "order": 1,
            "role": None,
        },
        {
            "box": [10, 10, 2, 1],
            "contents": "This is a test.",
            "direction": "vertical",
            "order": 1,
            "role": None,
        },
    ]

    paragraphs = [ParagraphSchema(**paragraph) for paragraph in paragraphs]
    assert judge_page_direction(paragraphs) == "vertical"


def test_extract_words_within_element():
    paragraph = {
        "box": [0, 0, 1, 1],
        "contents": "This is a test.",
        "direction": "horizontal",
        "order": 1,
        "role": None,
    }

    element = ParagraphSchema(**paragraph)

    words = [
        {
            "points": [[10, 10], [11, 10], [11, 11], [10, 11]],
            "content": "This",
            "direction": "horizontal",
            "rec_score": 0.9,
            "det_score": 0.9,
        }
    ]

    words = [WordPrediction(**word) for word in words]

    words, direction, checklist = extract_words_within_element(words, element)

    assert words is None
    assert direction is None
    assert checklist == [False]

    paragraph = {
        "box": [0, 0, 5, 5],
        "contents": "This is a test.",
        "direction": "horizontal",
        "order": 1,
        "role": None,
    }

    element = ParagraphSchema(**paragraph)

    words = [
        {
            "points": [[0, 0], [1, 0], [1, 1], [0, 1]],
            "content": "Hello",
            "direction": "horizontal",
            "rec_score": 0.9,
            "det_score": 0.9,
        },
        {
            "points": [[0, 1], [1, 1], [1, 2], [0, 2]],
            "content": "World",
            "direction": "horizontal",
            "rec_score": 0.9,
            "det_score": 0.9,
        },
    ]

    words = [WordPrediction(**word) for word in words]

    words, direction, checklist = extract_words_within_element(words, element)

    assert words == "Hello\nWorld"
    assert direction == "horizontal"
    assert checklist == [True, True]

    paragraph = {
        "box": [0, 0, 5, 5],
        "contents": "This is a test.",
        "direction": "horizontal",
        "order": 1,
        "role": None,
    }

    element = ParagraphSchema(**paragraph)

    words = [
        {
            "points": [[2, 0], [3, 0], [3, 1], [2, 1]],
            "content": "Hello",
            "direction": "vertical",
            "rec_score": 0.9,
            "det_score": 0.9,
        },
        {
            "points": [[0, 1], [1, 1], [1, 2], [0, 2]],
            "content": "World",
            "direction": "vertical",
            "rec_score": 0.9,
            "det_score": 0.9,
        },
    ]

    words = [WordPrediction(**word) for word in words]

    words, direction, checklist = extract_words_within_element(words, element)

    assert words == "Hello\nWorld"
    assert direction == "vertical"
    assert checklist == [True, True]


def test_is_vertical():
    quad = [[0, 0], [1, 0], [1, 1], [0, 1]]
    assert not is_vertical(quad)
    quad = [[0, 0], [1, 0], [1, 3], [0, 3]]
    assert is_vertical(quad)


def test_is_noise():
    quad = [[0, 0], [1, 0], [1, 1], [0, 1]]
    assert is_noise(quad)

    quad = [[0, 0], [20, 0], [20, 20], [0, 20]]
    assert not is_noise(quad)


def test_recursive_update():
    original = {"a": {"b": {"c": 1, "d": 2}}}
    update = {"a": {"b": {"d": 3, "e": 4}}}

    updated = recursive_update(original, update)

    assert updated == {"a": {"b": {"c": 1, "d": 3, "e": 4}}}


def test_extract_words_within_table():
    points = [
        [[0, 0], [3, 0], [3, 1], [0, 1]],
        [[3, 0], [5, 0], [5, 1], [3, 1]],
        [[0, 1], [1, 1], [1, 4], [0, 4]],
        [[3, 1], [3, 1], [4, 4], [4, 4]],
    ]

    scores = [0.9, 0.9, 0.9, 0.9]

    words = TextDetectorSchema(points=points, scores=scores)

    table = {
        "box": [0, 0, 3, 3],
        "n_row": 2,
        "n_col": 2,
        "rows": [],
        "cols": [],
        "cells": [],
        "order": 0,
    }

    table = TableStructureRecognizerSchema(**table)
    checklist = [False, False, False, False]
    h_words, v_words, checklist = _extract_words_within_table(words, table, checklist)

    assert len(h_words) == 1
    assert len(v_words) == 1
    assert checklist == [True, False, True, False]


def test_calc_overlap_words_on_lines():
    lines = [
        {
            "box": [0, 0, 2, 1],
            "score": 0.9,
        },
        {
            "box": [0, 1, 1, 1],
            "score": 0.9,
        },
    ]

    lines = [TableLineSchema(**line) for line in lines]

    words = [
        {
            "points": [[0, 0], [1, 0], [1, 1], [0, 1]],
        },
        {
            "points": [[1, 0], [3, 0], [3, 1], [1, 1]],
        },
    ]

    overrap_ratios = _calc_overlap_words_on_lines(lines, words)

    assert overrap_ratios == [[1.0, 0.0], [0.5, 0.0]]


def test_correct_vertical_word_boxes():
    words = [
        {
            "points": [[0, 0], [20, 0], [20, 100], [0, 100]],
            "score": 0.9,
        },
    ]

    cols = [TableLineSchema(box=[0, 0, 20, 100], score=0.9)]
    rows = [
        TableLineSchema(box=[0, 0, 20, 50], score=0.9),
        TableLineSchema(box=[0, 50, 20, 100], score=0.9),
    ]

    cells = [
        {
            "col": 1,
            "row": 1,
            "col_span": 1,
            "row_span": 1,
            "box": [0, 0, 20, 50],
            "contents": None,
        },
        {
            "col": 1,
            "row": 2,
            "col_span": 1,
            "row_span": 1,
            "box": [0, 50, 20, 100],
            "contents": None,
        },
    ]

    cells = [TableCellSchema(**cell) for cell in cells]

    table = {
        "box": [0, 0, 100, 20],
        "n_row": 2,
        "n_col": 1,
        "rows": rows,
        "cols": cols,
        "cells": cells,
        "order": 0,
    }

    table = TableStructureRecognizerSchema(**table)

    overrap_ratios = _calc_overlap_words_on_lines(cols, words)

    points, scores = _correct_vertical_word_boxes(
        overrap_ratios,
        table,
        words,
    )

    assert len(points) == 2
    assert len(scores) == 2
    assert points[0] == [[0, 0], [20, 0], [20, 50], [0, 50]]
    assert points[1] == [[0, 50], [20, 50], [20, 100], [0, 100]]


def test_correct_horizontal_word_boxes():
    words = [
        {
            "points": [[0, 0], [100, 0], [100, 20], [0, 20]],
            "score": 0.9,
        },
    ]

    cols = [
        TableLineSchema(box=[0, 0, 50, 20], score=0.9),
        TableLineSchema(box=[50, 0, 100, 20], score=0.9),
    ]
    rows = [
        TableLineSchema(box=[0, 0, 100, 20], score=0.9),
    ]

    cells = [
        {
            "col": 1,
            "row": 1,
            "col_span": 1,
            "row_span": 1,
            "box": [0, 0, 50, 20],
            "contents": None,
        },
        {
            "col": 2,
            "row": 1,
            "col_span": 1,
            "row_span": 1,
            "box": [50, 0, 100, 20],
            "contents": None,
        },
    ]

    cells = [TableCellSchema(**cell) for cell in cells]

    table = {
        "box": [0, 0, 20, 100],
        "n_row": 2,
        "n_col": 1,
        "rows": rows,
        "cols": cols,
        "cells": cells,
        "order": 0,
    }

    table = TableStructureRecognizerSchema(**table)

    overrap_ratios = _calc_overlap_words_on_lines(cols, words)

    points, scores = _correct_horizontal_word_boxes(
        overrap_ratios,
        table,
        words,
    )

    assert len(points) == 2
    assert len(scores) == 2
    assert points[0] == [[0, 0], [50, 0], [50, 20], [0, 20]]
    assert points[1] == [[50, 0], [100, 0], [100, 20], [50, 20]]


def test_split_text_across_cells():
    points = [
        [[0, 0], [100, 0], [100, 20], [0, 20]],
    ]

    scores = [0.9]

    words = TextDetectorSchema(points=points, scores=scores)

    cols = [
        TableLineSchema(box=[0, 0, 50, 20], score=0.9),
        TableLineSchema(box=[50, 0, 100, 20], score=0.9),
    ]
    rows = [
        TableLineSchema(box=[0, 0, 100, 20], score=0.9),
    ]

    cells = [
        {
            "col": 1,
            "row": 1,
            "col_span": 1,
            "row_span": 1,
            "box": [0, 0, 50, 20],
            "contents": None,
        },
        {
            "col": 2,
            "row": 1,
            "col_span": 1,
            "row_span": 1,
            "box": [50, 0, 100, 20],
            "contents": None,
        },
    ]

    cells = [TableCellSchema(**cell) for cell in cells]

    table = {
        "box": [0, 0, 100, 20],
        "n_row": 2,
        "n_col": 1,
        "rows": rows,
        "cols": cols,
        "cells": cells,
        "order": 0,
    }

    table = TableStructureRecognizerSchema(**table)

    Layout = DocumentAnalyzerSchema(
        paragraphs=[],
        figures=[],
        tables=[table],
        words=[],
    )

    results = _split_text_across_cells(words, Layout)

    assert len(results.points) == 2
    assert len(results.scores) == 2
    assert results.points[0] == [[0, 0], [50, 0], [50, 20], [0, 20]]
    assert results.points[1] == [[50, 0], [100, 0], [100, 20], [50, 20]]
