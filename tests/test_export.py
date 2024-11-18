import json
import os

from yomitoku.document_analyzer import DocumentAnalyzerSchema, ParagraphSchema
from yomitoku.export.export_csv import paragraph_to_csv, table_to_csv
from yomitoku.export.export_html import (
    convert_text_to_html,
    paragraph_to_html,
    table_to_html,
)
from yomitoku.export.export_json import paragraph_to_json, table_to_json
from yomitoku.export.export_markdown import (
    escape_markdown_special_chars,
    paragraph_to_md,
    table_to_md,
)
from yomitoku.export.utils import sort_elements
from yomitoku.layout_analyzer import LayoutAnalyzerSchema
from yomitoku.layout_parser import Element, LayoutParserSchema
from yomitoku.ocr import OCRSchema, WordPrediction
from yomitoku.table_structure_recognizer import (
    TableCellSchema,
    TableStructureRecognizerSchema,
)
from yomitoku.text_detector import TextDetectorSchema
from yomitoku.text_recognizer import TextRecognizerSchema


def test_convert_text_to_html():

    texts = [
        {
            "text": "これはテストです。<p>がんばりましょう。</p>",
            "expected": "これはテストです。&lt;p&gt;がんばりましょう。&lt;/p&gt;",
        },
        {
            "text": "これはテストです。https://www.google.com",
            "expected": "これはテストです。https://www.google.com",
        },
        {
            "text": "これはテストです。<a href='https://www.google.com'>Google</a>",
            "expected": "これはテストです。&lt;a href=&#x27;https://www.google.com&#x27;&gt;Google&lt;/a&gt;",
        },
    ]

    for text in texts:
        assert convert_text_to_html(text["text"]) == text["expected"]


def test_table_to_html():
    cells = [
        {
            "box": [0, 0, 10, 10],
            "col": 1,
            "row": 1,
            "row_span": 2,
            "col_span": 1,
            "contents": "dummy\n",
        },
        {
            "box": [0, 0, 10, 10],
            "row": 1,
            "col": 2,
            "row_span": 1,
            "col_span": 1,
            "contents": "dummy\n",
        },
        {
            "box": [0, 0, 10, 10],
            "row": 2,
            "col": 2,
            "row_span": 1,
            "col_span": 1,
            "contents": "",
        },
    ]
    cells = [TableCellSchema(**cell) for cell in cells]
    table = {"box": [0, 0, 100, 100], "n_row": 2, "n_col": 2, "cells": cells}
    table = TableStructureRecognizerSchema(**table)
    expected = '<table border="1" style="border-collapse: collapse"><tr><td rowspan="2" colspan="1">dummy<br></td><td rowspan="1" colspan="1">dummy<br></td></tr><tr><td rowspan="1" colspan="1"></td></tr></table>'
    assert table_to_html(table, ignore_line_break=False)["html"] == expected

    expected = '<table border="1" style="border-collapse: collapse"><tr><td rowspan="2" colspan="1">dummy</td><td rowspan="1" colspan="1">dummy</td></tr><tr><td rowspan="1" colspan="1"></td></tr></table>'
    assert table_to_html(table, ignore_line_break=True)["html"] == expected


def test_paragraph_to_html():
    paragraph = {
        "direction": "horizontal",
        "box": [0, 0, 10, 10],
        "contents": "これはテストです。<a href='https://www.google.com'>Google</a>\n",
    }

    paragraph = ParagraphSchema(**paragraph)
    expected = "<p>これはテストです。&lt;a href=&#x27;https://www.google.com&#x27;&gt;Google&lt;/a&gt;<br></p>"
    assert (
        paragraph_to_html(paragraph, ignore_line_break=False)["html"]
        == expected
    )

    expected = "<p>これはテストです。&lt;a href=&#x27;https://www.google.com&#x27;&gt;Google&lt;/a&gt;</p>"
    assert (
        paragraph_to_html(paragraph, ignore_line_break=True)["html"]
        == expected
    )


def test_sort_elements():
    elements = []
    directions = []

    assert sort_elements(elements, directions) == []
    elements = [
        {"box": [20, 10, 10, 10]},
        {"box": [10, 0, 10, 10]},
        {"box": [0, 20, 10, 10]},
    ]

    directions = ["horizontal"]
    expected = [
        {"box": [10, 0, 10, 10]},
        {"box": [20, 10, 10, 10]},
        {"box": [0, 20, 10, 10]},
    ]

    assert sort_elements(elements, directions) == expected

    directions = ["vertical"]
    expected = [
        {"box": [20, 10, 10, 10]},
        {"box": [10, 0, 10, 10]},
        {"box": [0, 20, 10, 10]},
    ]

    assert sort_elements(elements, directions) == expected


def test_escape_markdown_special_chars():
    texts = [
        {
            "text": "![image](https://www.google.com)",
            "expected": "\!\[image\]\(https://www\.google\.com\)",
        },
        {
            "text": "**これはテストです**",
            "expected": "\*\*これはテストです\*\*",
        },
        {
            "text": "- これはテストです",
            "expected": "\- これはテストです",
        },
        {
            "text": "1. これはテストです",
            "expected": "1\. これはテストです",
        },
        {
            "text": "| これはテストです",
            "expected": "\| これはテストです",
        },
        {
            "text": "```python\nprint('Hello, World!')\n```",
            "expected": "\`\`\`python\nprint\('Hello, World\!'\)\n\`\`\`",
        },
    ]

    for text in texts:
        assert escape_markdown_special_chars(text["text"]) == text["expected"]


def test_paragraph_to_md():
    paragraph = {
        "direction": "horizontal",
        "box": [0, 0, 10, 10],
        "contents": "print('Hello, World!')\n",
    }

    paragraph = ParagraphSchema(**paragraph)

    expected = "print\('Hello, World\!'\)<br>\n"
    assert (
        paragraph_to_md(paragraph, ignore_line_break=False)["md"] == expected
    )

    expected = "print\('Hello, World\!'\)\n"
    assert paragraph_to_md(paragraph, ignore_line_break=True)["md"] == expected


def test_table_to_md():
    cells = [
        {
            "box": [0, 0, 10, 10],
            "col": 1,
            "row": 1,
            "row_span": 2,
            "col_span": 1,
            "contents": "dummy\n",
        },
        {
            "box": [0, 0, 10, 10],
            "row": 1,
            "col": 2,
            "row_span": 1,
            "col_span": 1,
            "contents": "dummy\n",
        },
        {
            "box": [0, 0, 10, 10],
            "row": 2,
            "col": 2,
            "row_span": 1,
            "col_span": 1,
            "contents": "dummy\n",
        },
    ]
    cells = [TableCellSchema(**cell) for cell in cells]
    table = {"box": [0, 0, 100, 100], "n_row": 2, "n_col": 2, "cells": cells}
    table = TableStructureRecognizerSchema(**table)

    expected = "|dummy<br>|dummy<br>|\n|-|-|\n||dummy<br>|\n"

    assert table_to_md(table, ignore_line_break=False)["md"] == expected

    expected = "|dummy|dummy|\n|-|-|\n||dummy|\n"
    assert table_to_md(table, ignore_line_break=True)["md"] == expected


def test_table_to_csv():
    cells = [
        {
            "box": [0, 0, 10, 10],
            "col": 1,
            "row": 1,
            "row_span": 2,
            "col_span": 1,
            "contents": "dummy\n",
        },
        {
            "box": [0, 0, 10, 10],
            "row": 1,
            "col": 2,
            "row_span": 1,
            "col_span": 1,
            "contents": "dummy\n",
        },
        {
            "box": [0, 0, 10, 10],
            "row": 2,
            "col": 2,
            "row_span": 1,
            "col_span": 1,
            "contents": "dummy\n",
        },
    ]
    cells = [TableCellSchema(**cell) for cell in cells]
    table = {"box": [0, 0, 100, 100], "n_row": 2, "n_col": 2, "cells": cells}
    table = TableStructureRecognizerSchema(**table)

    expected = [["dummy\n", "dummy\n"], ["", "dummy\n"]]
    assert table_to_csv(table, ignore_line_break=False) == expected

    expected = [["dummy", "dummy"], ["", "dummy"]]
    assert table_to_csv(table, ignore_line_break=True) == expected


def test_paragraph_to_csv():
    paragraph = {
        "direction": "horizontal",
        "box": [0, 0, 10, 10],
        "contents": "dummy\n",
    }

    paragraph = ParagraphSchema(**paragraph)

    expected = "dummy\n"
    assert paragraph_to_csv(paragraph, ignore_line_break=False) == expected

    expected = "dummy"
    assert paragraph_to_csv(paragraph, ignore_line_break=True) == expected


def test_paragraph_to_json():
    paragraph = {
        "direction": "horizontal",
        "box": [0, 0, 10, 10],
        "contents": "dummy\n",
    }

    paragraph = ParagraphSchema(**paragraph)
    paragraph_to_json(paragraph, ignore_line_break=False)

    assert paragraph.contents == "dummy\n"

    paragraph_to_json(paragraph, ignore_line_break=True)
    assert paragraph.contents == "dummy"


def test_table_to_json():
    cells = [
        {
            "box": [0, 0, 10, 10],
            "col": 1,
            "row": 1,
            "row_span": 2,
            "col_span": 1,
            "contents": "dummy\n",
        },
        {
            "box": [0, 0, 10, 10],
            "row": 1,
            "col": 2,
            "row_span": 1,
            "col_span": 1,
            "contents": "dummy\n",
        },
        {
            "box": [0, 0, 10, 10],
            "row": 2,
            "col": 2,
            "row_span": 1,
            "col_span": 1,
            "contents": "dummy\n",
        },
    ]
    cells = [TableCellSchema(**cell) for cell in cells]
    table = {"box": [0, 0, 100, 100], "n_row": 2, "n_col": 2, "cells": cells}
    table = TableStructureRecognizerSchema(**table)

    table_to_json(table, ignore_line_break=False)
    for cell in table.cells:
        assert cell.contents == "dummy\n"

    table_to_json(table, ignore_line_break=True)
    for cell in table.cells:
        assert cell.contents == "dummy"


def test_export(tmp_path):
    text_recogition = {
        "contents": ["test"],
        "points": [[[0, 0], [10, 10], [20, 20], [30, 30]]],
        "scores": [0.9],
        "directions": ["horizontal"],
    }
    texts = TextRecognizerSchema(**text_recogition)
    out_path = tmp_path / "tr.json"
    texts.to_json(out_path)
    with open(out_path, "r") as f:
        assert json.load(f) == texts.dict()

    text_detection = {
        "points": [[[0, 0], [10, 10], [20, 20], [30, 30]]],
        "scores": [0.9],
    }
    texts = TextDetectorSchema(**text_detection)
    out_path = tmp_path / "td.json"
    texts.to_json(out_path)
    with open(out_path, "r") as f:
        assert json.load(f) == texts.dict()

    words = {
        "points": [[0, 0], [10, 10], [20, 20], [30, 30]],
        "content": "test",
        "direction": "horizontal",
        "det_score": 0.9,
        "rec_score": 0.9,
    }

    words = WordPrediction(**words)
    out_path = tmp_path / "words.json"
    words.to_json(out_path)
    with open(out_path, "r") as f:
        assert json.load(f) == words.dict()

    result = {"words": [words]}
    ocr = OCRSchema(**result)

    out_path = tmp_path / "ocr.yaml"
    ocr.to_json(out_path)
    with open(out_path, "r") as f:
        assert json.load(f) == ocr.dict()

    element = {
        "box": [0, 0, 10, 10],
        "score": 0.9,
    }
    element = Element(**element)
    out_path = tmp_path / "element.json"
    element.to_json(out_path)
    with open(out_path, "r") as f:
        assert json.load(f) == element.dict()

    layout_parser = {
        "paragraphs": [element],
        "tables": [element],
        "figures": [element],
    }

    layout_parser = LayoutParserSchema(**layout_parser)
    out_path = tmp_path / "layout_parser.json"
    layout_parser.to_json(out_path)
    with open(out_path, "r") as f:
        assert json.load(f) == layout_parser.dict()

    layout_parser.to_json(out_path, ignore_line_break=True)
    with open(out_path, "r") as f:
        assert json.load(f) == layout_parser.dict()

    table_cell = {
        "box": [0, 0, 10, 10],
        "col": 1,
        "row": 1,
        "row_span": 2,
        "col_span": 1,
        "contents": "dummy\n",
    }

    table_cell = TableCellSchema(**table_cell)
    out_path = tmp_path / "table_cell.json"
    table_cell.to_json(out_path)
    with open(out_path, "r") as f:
        assert json.load(f) == table_cell.dict()

    tsr = {
        "box": [0, 0, 100, 100],
        "n_row": 2,
        "n_col": 2,
        "cells": [table_cell],
    }

    tsr = TableStructureRecognizerSchema(**tsr)
    out_path = tmp_path / "tsr.json"
    tsr.to_json(out_path)
    with open(out_path, "r") as f:
        assert json.load(f) == tsr.dict()

    layout_analyzer = {
        "paragraphs": [element],
        "tables": [tsr],
        "figures": [element],
    }

    layout_analyzer = LayoutAnalyzerSchema(**layout_analyzer)
    out_path = tmp_path / "layout_analyzer.json"
    layout_analyzer.to_json(out_path)
    with open(out_path, "r") as f:
        assert json.load(f) == layout_analyzer.dict()

    paragraph = {
        "direction": "horizontal",
        "box": [0, 0, 10, 10],
        "contents": "dummy\n",
    }
    paragraph = ParagraphSchema(**paragraph)
    out_path = tmp_path / "paragraph.json"
    paragraph.to_json(out_path)
    with open(out_path, "r") as f:
        assert json.load(f) == paragraph.dict()

    document_analyzer = {
        "paragraphs": [paragraph],
        "tables": [tsr],
        "figures": [element],
        "words": [words],
    }

    document_analyzer = DocumentAnalyzerSchema(**document_analyzer)
    out_path = tmp_path / "document_analyzer.json"
    document_analyzer.to_json(out_path)
    with open(out_path, "r") as f:
        assert json.load(f) == document_analyzer.dict()

    document_analyzer.to_csv(tmp_path / "document_analyzer.csv")
    document_analyzer.to_html(tmp_path / "document_analyzer.html")
    document_analyzer.to_markdown(tmp_path / "document_analyzer.md")

    assert os.path.exists(tmp_path / "document_analyzer.csv")
    assert os.path.exists(tmp_path / "document_analyzer.html")
    assert os.path.exists(tmp_path / "document_analyzer.md")
