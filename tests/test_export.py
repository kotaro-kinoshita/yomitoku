import os
import json


import numpy as np

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
from yomitoku.export.export_pydict import (
    export_to_dict,
    save_to_dict,
    convert_to_dict,
)

from yomitoku.schemas import (
    DocumentAnalyzerSchema,
    LayoutAnalyzerSchema,
    LayoutParserSchema,
    OCRSchema,
    ParagraphSchema,
    FigureSchema,
    TableCellSchema,
    TableLineSchema,
    TableStructureRecognizerSchema,
    TextDetectorSchema,
    TextRecognizerSchema,
    WordPrediction,
    Element,
)


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

    spans = [
        {
            "box": [0, 0, 10, 10],
            "score": 0.9,
        }
    ]

    rows = [TableLineSchema(**row) for row in rows]
    cols = [TableLineSchema(**col) for col in cols]
    spans = [TableLineSchema(**span) for span in spans]

    table = {
        "box": [0, 0, 100, 100],
        "n_row": 2,
        "n_col": 2,
        "cells": cells,
        "order": 0,
        "rows": rows,
        "cols": cols,
        "spans": spans,
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

    spans = [
        {
            "box": [0, 0, 10, 10],
            "score": 0.9,
        }
    ]

    rows = [TableLineSchema(**row) for row in rows]
    cols = [TableLineSchema(**col) for col in cols]
    spans = [TableLineSchema(**span) for span in spans]

    table = {
        "box": [0, 0, 100, 100],
        "n_row": 2,
        "n_col": 2,
        "cells": cells,
        "order": 0,
        "spans": spans,
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

    spans = [
        {
            "box": [0, 0, 10, 10],
            "score": 0.9,
        }
    ]

    rows = [TableLineSchema(**row) for row in rows]
    cols = [TableLineSchema(**col) for col in cols]
    spans = [TableLineSchema(**span) for span in spans]

    table = {
        "box": [0, 0, 100, 100],
        "n_row": 2,
        "n_col": 2,
        "cells": cells,
        "order": 0,
        "spans": spans,
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

    spans = [
        {
            "box": [0, 0, 10, 10],
            "score": 0.9,
        }
    ]

    rows = [TableLineSchema(**row) for row in rows]
    cols = [TableLineSchema(**col) for col in cols]
    spans = [TableLineSchema(**span) for span in spans]

    table = {
        "box": [0, 0, 100, 100],
        "n_row": 2,
        "n_col": 2,
        "cells": cells,
        "order": 0,
        "rows": rows,
        "cols": cols,
        "spans": spans,
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
    ocr.to_json(out_path)

    with open(out_path, "r") as f:
        assert json.load(f) == ocr.model_dump()

    element = {"box": [0, 0, 10, 10], "score": 0.9, "role": None}
    element = Element(**element)
    out_path = tmp_path / "element.json"
    element.to_json(out_path)

    with open(out_path, "r") as f:
        assert json.load(f) == element.model_dump()

    layout_parser = {
        "paragraphs": [element],
        "tables": [element],
        "figures": [element],
    }

    layout_parser = LayoutParserSchema(**layout_parser)
    out_path = tmp_path / "layout_parser.json"
    layout_parser.to_json(out_path)

    with open(out_path, "r") as f:
        assert json.load(f) == layout_parser.model_dump()

    layout_parser.to_json(out_path, ignore_line_break=True)
    with open(out_path, "r") as f:
        assert json.load(f) == layout_parser.model_dump()

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

    spans = [
        {
            "box": [0, 0, 10, 10],
            "score": 0.9,
        }
    ]

    rows = [TableLineSchema(**row) for row in rows]
    cols = [TableLineSchema(**col) for col in cols]
    spans = [TableLineSchema(**span) for span in spans]

    table_cell = TableCellSchema(**table_cell)
    out_path = tmp_path / "table_cell.json"
    table_cell.to_json(out_path)
    with open(out_path, "r") as f:
        assert json.load(f) == table_cell.model_dump()

    tsr = {
        "box": [0, 0, 100, 100],
        "n_row": 2,
        "n_col": 2,
        "cells": [table_cell],
        "order": 0,
        "rows": rows,
        "cols": cols,
        "spans": spans,
    }

    tsr = TableStructureRecognizerSchema(**tsr)
    out_path = tmp_path / "tsr.json"
    tsr.to_json(out_path)
    with open(out_path, "r") as f:
        assert json.load(f) == tsr.model_dump()

    layout_analyzer = {
        "paragraphs": [element],
        "tables": [tsr],
        "figures": [element],
    }

    layout_analyzer = LayoutAnalyzerSchema(**layout_analyzer)
    out_path = tmp_path / "layout_analyzer.json"
    layout_analyzer.to_json(out_path)
    with open(out_path, "r") as f:
        assert json.load(f) == layout_analyzer.model_dump()

    paragraph = {
        "direction": "horizontal",
        "box": [0, 0, 10, 10],
        "contents": "dummy\n",
        "order": 0,
        "role": None,
    }
    paragraph = ParagraphSchema(**paragraph)
    out_path = tmp_path / "paragraph.json"
    paragraph.to_json(out_path)
    with open(out_path, "r") as f:
        assert json.load(f) == paragraph.model_dump()

    figure = {
        "direction": "horizontal",
        "box": [0, 0, 10, 10],
        "paragraphs": [paragraph],
        "order": 0,
    }
    figure = FigureSchema(**figure)
    out_path = tmp_path / "figure.json"
    figure.to_json(out_path)
    with open(out_path, "r") as f:
        assert json.load(f) == figure.model_dump()

    document_analyzer = {
        "paragraphs": [paragraph],
        "tables": [tsr],
        "figures": [figure],
        "words": [words],
    }

    img = np.zeros((100, 100, 3), dtype=np.uint8)

    document_analyzer = DocumentAnalyzerSchema(**document_analyzer)
    out_path = tmp_path / "document_analyzer.json"
    document_analyzer.to_json(out_path)
    with open(out_path, "r") as f:
        assert json.load(f) == document_analyzer.model_dump()

    document_analyzer.to_csv(tmp_path / "document_analyzer.csv", img=img)
    document_analyzer.to_html(tmp_path / "document_analyzer.html", img=img)
    document_analyzer.to_markdown(tmp_path / "document_analyzer.md", img=img)

    # Test dict export
    pydict_result = document_analyzer.to_dict(include_figures=True)
    assert isinstance(pydict_result, dict)
    assert len(pydict_result["paragraphs"]) == 1
    assert len(pydict_result["tables"]) == 1
    assert len(pydict_result["figures"]) == 1

    # Test dict file export
    pydict_path = tmp_path / "document_analyzer.py"
    export_to_dict(document_analyzer, str(pydict_path), include_figures=True)
    assert pydict_path.exists()

    assert os.path.exists(tmp_path / "document_analyzer.csv")
    assert os.path.exists(tmp_path / "document_analyzer.html")
    assert os.path.exists(tmp_path / "document_analyzer.md")


def test_convert_to_dict():
    """Test convert_to_dict function with comprehensive scenarios"""
    # Create test data
    paragraph = ParagraphSchema(
        box=[0, 0, 10, 10], contents="テストパラグラフ", direction="horizontal", order=0, role="section_headings"
    )

    cell = TableCellSchema(row=0, col=0, row_span=1, col_span=1, box=[0, 0, 5, 5], contents="セル内容")

    tsr = TableStructureRecognizerSchema(
        box=[0, 0, 10, 10], n_row=1, n_col=1, rows=[], cols=[], spans=[], cells=[cell], order=1
    )

    figure = FigureSchema(direction="horizontal", box=[0, 0, 10, 10], paragraphs=[paragraph], order=2)

    words = WordPrediction(
        points=[[0, 0], [10, 0], [10, 10], [0, 10]],
        content="テスト",
        direction="horizontal",
        rec_score=0.9,
        det_score=0.8,
    )

    document_analyzer = DocumentAnalyzerSchema(paragraphs=[paragraph], tables=[tsr], figures=[figure], words=[words])

    # Test convert_to_dict without figures
    result = convert_to_dict(document_analyzer, include_figures=False)

    # Basic structure assertions
    assert isinstance(result, dict)
    assert set(result.keys()) == {"paragraphs", "tables", "figures"}

    # Content assertions
    assert len(result["paragraphs"]) == 1
    assert len(result["tables"]) == 1
    assert len(result["figures"]) == 0  # Should be empty when include_figures=False

    # Detailed content verification
    assert result["paragraphs"][0]["contents"] == "テストパラグラフ"
    assert result["paragraphs"][0]["order"] == 0
    assert result["paragraphs"][0]["role"] == "section_headings"

    assert result["tables"][0]["n_row"] == 1
    assert result["tables"][0]["n_col"] == 1
    assert len(result["tables"][0]["cells"]) == 1
    assert result["tables"][0]["cells"][0]["contents"] == "セル内容"

    # Test convert_to_dict with figures
    result_with_figures = convert_to_dict(document_analyzer, include_figures=True)

    assert len(result_with_figures["figures"]) == 1
    assert result_with_figures["figures"][0]["order"] == 2
    assert len(result_with_figures["figures"][0]["paragraphs"]) == 1
    assert result_with_figures["figures"][0]["paragraphs"][0]["contents"] == "テストパラグラフ"

    # Test empty input
    empty_document = DocumentAnalyzerSchema(paragraphs=[], tables=[], figures=[], words=[])
    empty_result = convert_to_dict(empty_document, include_figures=True)
    assert all(len(empty_result[key]) == 0 for key in ["paragraphs", "tables", "figures"])


def test_save_to_dict(tmp_path):
    """Test save_to_dict function with file operations and validation"""
    # Test data with various data types
    test_data = {
        "paragraphs": [{"contents": "テスト内容\n改行あり", "box": [0, 0, 10, 10], "order": 0, "role": "body_text"}],
        "tables": [
            {
                "n_row": 2,
                "n_col": 2,
                "cells": [{"row": 0, "col": 0, "contents": "ヘッダー"}, {"row": 1, "col": 0, "contents": "データ"}],
            }
        ],
        "figures": [],
    }

    # Test basic functionality
    output_file = tmp_path / "test_output.py"
    save_to_dict(test_data, str(output_file))

    # Verify file was created
    assert output_file.exists()
    assert output_file.is_file()
    assert output_file.stat().st_size > 0

    # Verify file contents
    content = output_file.read_text(encoding="utf-8")
    assert "yomitoku_result" in content
    assert "Generated by yomitoku export_to_dict" in content
    assert "テスト内容" in content
    assert "改行あり" in content

    # Verify the file contains valid, executable Python code
    exec_globals = {}
    exec(content, exec_globals)
    assert "yomitoku_result" in exec_globals
    assert exec_globals["yomitoku_result"] == test_data

    # Test with .py extension auto-addition
    output_without_ext = tmp_path / "test_no_ext"
    save_to_dict(test_data, str(output_without_ext))
    expected_py_file = tmp_path / "test_no_ext.py"
    assert expected_py_file.exists()

    # Test with directory creation
    nested_output = tmp_path / "nested" / "dir" / "output.py"
    save_to_dict(test_data, str(nested_output))
    assert nested_output.exists()
    assert nested_output.parent.exists()


def test_export_to_dict(tmp_path):
    """Test export_to_dict function with file output and return value validation"""
    # Create comprehensive test data
    paragraph = ParagraphSchema(
        box=[0, 0, 10, 10], contents="テストパラグラフ", direction="horizontal", order=0, role="section_headings"
    )

    cell = TableCellSchema(row=0, col=0, row_span=1, col_span=1, box=[0, 0, 5, 5], contents="セル内容")

    tsr = TableStructureRecognizerSchema(
        box=[0, 0, 10, 10], n_row=1, n_col=1, rows=[], cols=[], spans=[], cells=[cell], order=1
    )

    words = WordPrediction(
        points=[[0, 0], [10, 0], [10, 10], [0, 10]],
        content="テスト",
        direction="horizontal",
        rec_score=0.9,
        det_score=0.8,
    )

    document_analyzer = DocumentAnalyzerSchema(
        paragraphs=[paragraph], tables=[tsr], figures=[], words=[words]  # Empty figures for this test
    )

    # Test export_to_dict with file output
    output_file = tmp_path / "test_export.py"
    result = export_to_dict(document_analyzer, str(output_file), include_figures=True)

    # Verify return value structure
    assert isinstance(result, dict)
    assert set(result.keys()) == {"paragraphs", "tables", "figures"}
    assert len(result["paragraphs"]) == 1
    assert len(result["tables"]) == 1
    assert len(result["figures"]) == 0  # No figures in test data

    # Verify return value content
    assert result["paragraphs"][0]["contents"] == "テストパラグラフ"
    assert result["tables"][0]["cells"][0]["contents"] == "セル内容"

    # Verify file was created
    assert output_file.exists()

    # Verify file content can be imported
    content = output_file.read_text(encoding="utf-8")
    exec_globals = {}
    exec(content, exec_globals)
    assert "yomitoku_result" in exec_globals
    assert exec_globals["yomitoku_result"] == result

    # Test export_to_dict without file output (return only)
    result_no_file = export_to_dict(document_analyzer, out_path=None, include_figures=True)
    assert isinstance(result_no_file, dict)
    assert result_no_file == result  # Should be identical data

    # Test with different include_figures settings
    result_no_figures = export_to_dict(document_analyzer, out_path=None, include_figures=False)
    result_with_figures = export_to_dict(document_analyzer, out_path=None, include_figures=True)
    assert result_no_figures["figures"] == result_with_figures["figures"]  # Both should be empty in this case


def test_to_dict_method():
    """Test DocumentAnalyzerSchema.to_dict() method with comprehensive validation"""
    # Create test data with multiple elements
    paragraph1 = ParagraphSchema(
        box=[0, 0, 10, 10], contents="第1段落", direction="horizontal", order=0, role="section_headings"
    )

    paragraph2 = ParagraphSchema(
        box=[0, 15, 20, 25], contents="第2段落", direction="horizontal", order=1, role="body_text"
    )

    words = WordPrediction(
        points=[[0, 0], [10, 0], [10, 10], [0, 10]],
        content="テストワード",
        direction="horizontal",
        rec_score=0.95,
        det_score=0.87,
    )

    # Test with figures
    figure = FigureSchema(direction="horizontal", box=[0, 30, 15, 40], paragraphs=[paragraph1], order=2)

    document_analyzer = DocumentAnalyzerSchema(
        paragraphs=[paragraph1, paragraph2], tables=[], figures=[figure], words=[words]
    )

    # Test to_dict method without figures
    result_no_figures = document_analyzer.to_dict(include_figures=False)
    assert isinstance(result_no_figures, dict)
    assert set(result_no_figures.keys()) == {"paragraphs", "tables", "figures"}
    assert len(result_no_figures["paragraphs"]) == 2
    assert len(result_no_figures["tables"]) == 0
    assert len(result_no_figures["figures"]) == 0  # Should be empty

    # Verify paragraph content
    assert result_no_figures["paragraphs"][0]["contents"] == "第1段落"
    assert result_no_figures["paragraphs"][1]["contents"] == "第2段落"
    assert result_no_figures["paragraphs"][0]["order"] == 0
    assert result_no_figures["paragraphs"][1]["order"] == 1

    # Test to_dict method with figures
    result_with_figures = document_analyzer.to_dict(include_figures=True)
    assert len(result_with_figures["figures"]) == 1
    assert result_with_figures["figures"][0]["order"] == 2
    assert len(result_with_figures["figures"][0]["paragraphs"]) == 1
    assert result_with_figures["figures"][0]["paragraphs"][0]["contents"] == "第1段落"

    # Verify that paragraphs and tables are the same in both results
    assert result_no_figures["paragraphs"] == result_with_figures["paragraphs"]
    assert result_no_figures["tables"] == result_with_figures["tables"]

    # Test with empty document
    empty_document = DocumentAnalyzerSchema(paragraphs=[], tables=[], figures=[], words=[])
    empty_result = empty_document.to_dict(include_figures=True)
    assert all(len(empty_result[key]) == 0 for key in ["paragraphs", "tables", "figures"])
