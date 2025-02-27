import os

import numpy as np

from yomitoku.document_analyzer import (
    DocumentAnalyzerSchema,
    ParagraphSchema,
    FigureSchema,
)
from yomitoku.export.export_csv import paragraph_to_csv, table_to_csv, save_csv
from yomitoku.export.export_html import (
    convert_text_to_html,
    paragraph_to_html,
    table_to_html,
    save_html,
)
from yomitoku.export.export_json import paragraph_to_json, table_to_json
from yomitoku.export.export_markdown import (
    escape_markdown_special_chars,
    paragraph_to_md,
    table_to_md,
    save_markdown,
)
from yomitoku.layout_analyzer import LayoutAnalyzerSchema
from yomitoku.layout_parser import Element, LayoutParserSchema
from yomitoku.ocr import OCRSchema, WordPrediction
from yomitoku.table_structure_recognizer import (
    TableCellSchema,
    TableLineSchema,
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

    rows = [
        {
            "box": [0, 0, 10, 10],
            "score": 0.9,
        }
    ]

    cols = [
        {
            "box": [0, 0, 10, 10],
            "score": 0.9,
        }
    ]

    rows = [TableLineSchema(**row) for row in rows]
    cols = [TableLineSchema(**col) for col in cols]

    table = {
        "box": [0, 0, 100, 100],
        "n_row": 2,
        "n_col": 2,
        "cells": cells,
        "order": 0,
        "rows": rows,
        "cols": cols,
    }

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
        "order": 0,
        "role": None,
    }

    paragraph = ParagraphSchema(**paragraph)
    expected = "<p>これはテストです。&lt;a href=&#x27;https://www.google.com&#x27;&gt;Google&lt;/a&gt;<br></p>"
    assert paragraph_to_html(paragraph, ignore_line_break=False)["html"] == expected

    expected = "<p>これはテストです。&lt;a href=&#x27;https://www.google.com&#x27;&gt;Google&lt;/a&gt;</p>"
    assert paragraph_to_html(paragraph, ignore_line_break=True)["html"] == expected


def test_escape_markdown_special_chars():
    texts = [
        {
            "text": "![image](https://www.google.com)",
            "expected": "\!\[image\]\(https://www.google.com\)",
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
            "expected": "1. これはテストです",
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
        "order": 0,
        "role": None,
    }

    paragraph = ParagraphSchema(**paragraph)

    expected = "print\('Hello, World\!'\)<br>\n"
    assert paragraph_to_md(paragraph, ignore_line_break=False)["md"] == expected

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

    rows = [
        {
            "box": [0, 0, 10, 10],
            "score": 0.9,
        }
    ]

    cols = [
        {
            "box": [0, 0, 10, 10],
            "score": 0.9,
        }
    ]

    rows = [TableLineSchema(**row) for row in rows]
    cols = [TableLineSchema(**col) for col in cols]

    table = {
        "box": [0, 0, 100, 100],
        "n_row": 2,
        "n_col": 2,
        "cells": cells,
        "order": 0,
        "rows": rows,
        "cols": cols,
    }
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

    rows = [
        {
            "box": [0, 0, 10, 10],
            "score": 0.9,
        }
    ]

    cols = [
        {
            "box": [0, 0, 10, 10],
            "score": 0.9,
        }
    ]

    rows = [TableLineSchema(**row) for row in rows]
    cols = [TableLineSchema(**col) for col in cols]

    table = {
        "box": [0, 0, 100, 100],
        "n_row": 2,
        "n_col": 2,
        "cells": cells,
        "order": 0,
        "rows": rows,
        "cols": cols,
    }
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
        "order": 0,
        "role": None,
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
        "order": 0,
        "role": None,
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

    rows = [
        {
            "box": [0, 0, 10, 10],
            "score": 0.9,
        }
    ]

    cols = [
        {
            "box": [0, 0, 10, 10],
            "score": 0.9,
        }
    ]

    rows = [TableLineSchema(**row) for row in rows]
    cols = [TableLineSchema(**col) for col in cols]

    table = {
        "box": [0, 0, 100, 100],
        "n_row": 2,
        "n_col": 2,
        "cells": cells,
        "order": 0,
        "rows": rows,
        "cols": cols,
    }
    table = TableStructureRecognizerSchema(**table)

    table_to_json(table, ignore_line_break=False)
    # for cell in table.cells:
    #    assert cell.contents == "dummy\n"

    table_to_json(table, ignore_line_break=True)
    # for cell in table.cells:
    #    assert cell.contents == "dummy"


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

    text_detection = {
        "points": [[[0, 0], [10, 10], [20, 20], [30, 30]]],
        "scores": [0.9],
    }
    texts = TextDetectorSchema(**text_detection)
    out_path = tmp_path / "td.json"
    texts.to_json(out_path)

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

    result = {"words": [words]}
    ocr = OCRSchema(**result)

    out_path = tmp_path / "ocr.yaml"
    json = ocr.to_json(out_path)
    assert json == ocr.model_dump()

    element = {"box": [0, 0, 10, 10], "score": 0.9, "role": None}
    element = Element(**element)
    out_path = tmp_path / "element.json"
    json = element.to_json(out_path)
    assert json == element.model_dump()

    layout_parser = {
        "paragraphs": [element],
        "tables": [element],
        "figures": [element],
    }

    layout_parser = LayoutParserSchema(**layout_parser)
    out_path = tmp_path / "layout_parser.json"
    json = layout_parser.to_json(out_path)

    # with open(out_path, "r") as f:
    assert json == layout_parser.model_dump()

    json = layout_parser.to_json(out_path, ignore_line_break=True)
    assert json == layout_parser.model_dump()
    # with open(out_path, "r") as f:
    #    assert json.load(f) == layout_parser.model_dump()

    table_cell = {
        "box": [0, 0, 10, 10],
        "col": 1,
        "row": 1,
        "row_span": 2,
        "col_span": 1,
        "contents": "dummy\n",
    }

    rows = [
        {
            "box": [0, 0, 10, 10],
            "score": 0.9,
        }
    ]

    cols = [
        {
            "box": [0, 0, 10, 10],
            "score": 0.9,
        }
    ]

    rows = [TableLineSchema(**row) for row in rows]
    cols = [TableLineSchema(**col) for col in cols]

    table_cell = TableCellSchema(**table_cell)
    out_path = tmp_path / "table_cell.json"
    json = table_cell.to_json(out_path)
    assert json == table_cell.model_dump()
    # with open(out_path, "r") as f:
    #    assert json.load(f) == table_cell.model_dump()

    tsr = {
        "box": [0, 0, 100, 100],
        "n_row": 2,
        "n_col": 2,
        "cells": [table_cell],
        "order": 0,
        "rows": rows,
        "cols": cols,
    }

    tsr = TableStructureRecognizerSchema(**tsr)
    out_path = tmp_path / "tsr.json"
    json = tsr.to_json(out_path)
    assert json == tsr.model_dump()
    # with open(out_path, "r") as f:
    #    assert json.load(f) == tsr.model_dump()

    layout_analyzer = {
        "paragraphs": [element],
        "tables": [tsr],
        "figures": [element],
    }

    layout_analyzer = LayoutAnalyzerSchema(**layout_analyzer)
    out_path = tmp_path / "layout_analyzer.json"
    json = layout_analyzer.to_json(out_path)
    assert json == layout_analyzer.model_dump()
    # with open(out_path, "r") as f:
    #    assert json.load(f) == layout_analyzer.model_dump()

    paragraph = {
        "direction": "horizontal",
        "box": [0, 0, 10, 10],
        "contents": "dummy\n",
        "order": 0,
        "role": None,
    }
    paragraph = ParagraphSchema(**paragraph)
    out_path = tmp_path / "paragraph.json"
    json = paragraph.to_json(out_path)
    assert json == paragraph.model_dump()
    # with open(out_path, "r") as f:
    #    assert json.load(f) == paragraph.model_dump()

    figure = {
        "direction": "horizontal",
        "box": [0, 0, 10, 10],
        "paragraphs": [paragraph],
        "order": 0,
    }
    figure = FigureSchema(**figure)
    out_path = tmp_path / "figure.json"
    json = figure.to_json(out_path)
    assert json == figure.model_dump()
    # with open(out_path, "r") as f:
    #    assert json.load(f) == figure.model_dump()

    document_analyzer = {
        "paragraphs": [paragraph],
        "tables": [tsr],
        "figures": [figure],
        "words": [words],
    }

    img = np.zeros((100, 100, 3), dtype=np.uint8)

    document_analyzer = DocumentAnalyzerSchema(**document_analyzer)
    out_path = tmp_path / "document_analyzer.json"
    json = document_analyzer.to_json(out_path)
    assert json == document_analyzer.model_dump()
    # with open(out_path, "r") as f:
    #    assert json.load(f) == document_analyzer.model_dump()

    csv = document_analyzer.to_csv(tmp_path / "document_analyzer.csv", img=img)
    html = document_analyzer.to_html(tmp_path / "document_analyzer.html", img=img)
    md = document_analyzer.to_markdown(tmp_path / "document_analyzer.md", img=img)

    save_csv(tmp_path / "document_analyzer.csv", "utf-8", csv)
    save_html(tmp_path / "document_analyzer.html", "utf-8", html)
    save_markdown(tmp_path / "document_analyzer.md", "utf-8", md)

    assert os.path.exists(tmp_path / "document_analyzer.csv")
    assert os.path.exists(tmp_path / "document_analyzer.html")
    assert os.path.exists(tmp_path / "document_analyzer.md")
