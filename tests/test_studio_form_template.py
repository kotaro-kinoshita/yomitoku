from __future__ import annotations

import pytest
from pydantic import ValidationError

from yomitoku.schemas.document_analyzer import WordPrediction
from yomitoku.studio_form_template import (
    StudioFormTemplate,
    apply_studio_form_template,
    load_studio_form_template,
)


def _template():
    return StudioFormTemplate.model_validate(
        {
            "kind": "form-template",
            "version": 3,
            "fields": [],
            "structure": {
                "tables": [
                    {
                        "id": "t0",
                        "normBox": [0.1, 0.2, 0.9, 0.8],
                        "style": "border",
                        "cells": [
                            {
                                "id": "r0c0",
                                "normBox": [0.1, 0.2, 0.4, 0.4],
                                "role": "header",
                                "contents": "氏名",
                            },
                            {
                                "id": "r0c1",
                                "normBox": [0.4, 0.2, 0.9, 0.4],
                                "role": "cell",
                                "contents": "作成元の値",
                            },
                        ],
                        "kvItems": [{"id": "kv0", "key": ["r0c0"], "value": "r0c1"}],
                        "grids": [
                            {
                                "id": "g0",
                                "normBox": [0.1, 0.2, 0.9, 0.4],
                                "nRow": 1,
                                "nCol": 2,
                                "colHeaders": [["r0c0"], []],
                                "data": [["r0c0", "r0c1"]],
                            }
                        ],
                    }
                ],
                "paragraphs": [
                    {
                        "id": "p0",
                        "normBox": [0.1, 0.05, 0.9, 0.15],
                        "role": "section_headings",
                        "contents": "申込書",
                    }
                ],
            },
        }
    )


def _word(content, points):
    return WordPrediction(
        content=content,
        points=points,
        direction="horizontal",
        rec_score=1.0,
        det_score=1.0,
    )


def test_load_studio_template(tmp_path):
    path = tmp_path / "table.template.json"
    path.write_text(_template().model_dump_json(by_alias=True), encoding="utf-8")

    template = load_studio_form_template(path)
    assert template.kind == "form-template"
    assert template.version == 3
    assert len(template.structure.tables[0].cells) == 2
    result = apply_studio_form_template(template, [], width=100, height=100)
    assert len(result.to_structured().tables) == 1


def test_apply_uses_template_structure_and_current_ocr_only():
    words = [
        _word("申込書", [[20, 6], [80, 6], [80, 14], [20, 14]]),
        _word("山田太郎", [[45, 22], [85, 22], [85, 38], [45, 38]]),
    ]
    result = apply_studio_form_template(_template(), words, width=100, height=100)

    table = result.tables[0]
    assert table.box == [10, 20, 90, 80]
    assert table.cells["r0c0"].contents == "氏名"  # label fallback
    assert table.cells["r0c1"].contents == "山田太郎"
    assert table.cells["r0c1"].contents != "作成元の値"
    assert table.cells["r0c1"].meta["fromTemplate"] is True
    assert table.kv_items[0].value == "r0c1"
    assert table.grids[0].n_row == 1
    assert result.paragraphs[0].contents == "申込書"


def test_apply_never_fills_empty_value_from_template():
    result = apply_studio_form_template(_template(), [], width=100, height=100)
    assert result.tables[0].cells["r0c0"].contents == "氏名"
    assert result.tables[0].cells["r0c1"].contents == ""


def test_accepts_studio_template_v2():
    data = _template().model_dump(by_alias=True)
    data["version"] = 2
    assert StudioFormTemplate.model_validate(data).version == 2


def test_rejects_unknown_cell_reference():
    data = _template().model_dump(by_alias=True)
    data["structure"]["tables"][0]["kvItems"][0]["value"] = "missing"
    with pytest.raises(ValidationError, match="unknown cells"):
        StudioFormTemplate.model_validate(data)
