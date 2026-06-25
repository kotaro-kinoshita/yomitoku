# tests/test_table_semantic_parser_with_table_detector_schema.py

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pytest

from yomitoku.table_semantic_parser import (
    TableSemanticParser,
)
from yomitoku.ocr import OCRSchema

from yomitoku.schemas.table_semantic_parser import (
    TableDetectorSchema,
)

from collections import Counter
from typing import Any, Dict, List, Tuple


def load_table_detector_list(json_path: str | Path) -> List[TableDetectorSchema]:
    p = Path(json_path)
    payload = json.loads(p.read_text(encoding="utf-8"))

    if not isinstance(payload, list):
        raise ValueError(f"Expected list at root: {p}")

    tables: List[TableDetectorSchema] = []
    for i, t in enumerate(payload):
        try:
            tables.append(TableDetectorSchema.model_validate(t))
        except Exception as e:
            raise ValueError(f"Invalid table payload at index={i} file={p}") from e

    return tables


def dump_schema(obj):
    return obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()


def canonicalize_for_compare(x):
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            out[k] = canonicalize_for_compare(v)
        return out
    if isinstance(x, list):
        return [canonicalize_for_compare(v) for v in x]
    return x


def extract_metrics(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    doc: TableSemanticParserSchema.model_dump() 相当のdict
    """
    tables = doc.get("tables", []) or []

    table_metrics: List[Dict[str, Any]] = []
    for t in tables:
        grids = t.get("grids", []) or []
        kvs = t.get("kv_items", []) or []

        shapes = []
        for g in grids:
            shapes.append((int(g.get("n_row", 0)), int(g.get("n_col", 0))))

        # gridの順序が変わっても良いように Counter で保持
        table_metrics.append(
            {
                "n_grids": len(grids),
                "n_kv_items": len(kvs),
                "grid_shapes": Counter(shapes),
            }
        )

    return {
        "n_tables": len(tables),
        "tables": table_metrics,
    }


def normalize_table_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    テーブルの順序が変わっても比較できるように正規化する
    - (n_grids, n_kv_items, grid_shapes) をキーにソート
    """

    def key(
        t: Dict[str, Any],
    ) -> Tuple[int, int, Tuple[Tuple[Tuple[int, int], int], ...]]:
        shapes = tuple(sorted(((k[0], k[1]), v) for k, v in t["grid_shapes"].items()))
        return (t["n_grids"], t["n_kv_items"], shapes)

    tables = sorted(metrics["tables"], key=key)
    return {"n_tables": metrics["n_tables"], "tables": tables}


def assert_metrics_equal(got_doc: Dict[str, Any], exp_doc: Dict[str, Any]) -> None:
    got = normalize_table_metrics(extract_metrics(got_doc))
    exp = normalize_table_metrics(extract_metrics(exp_doc))
    assert got == exp


@pytest.fixture
def parser():
    # values() を踏まないように visualize=False にする
    p = TableSemanticParser(configs={}, device="cpu", visualize=False)
    p.grid_only = False
    p.merge_same_column_values = False
    return p


def test_semantic_output_matches_golden(parser, monkeypatch):
    input_dir = Path("tests/data/table_semantic_inputs")  # 10 json
    golden_dir = Path("tests/data/table_semantic_outputs")  # 期待出力
    golden_dir.mkdir(parents=True, exist_ok=True)

    img = np.zeros((32, 32, 3), dtype=np.uint8)

    files = sorted(input_dir.glob("*.json"))
    assert len(files) == 10

    for jf in files:
        tables = load_table_detector_list(jf)
        empty_ocr = OCRSchema(words=[])

        async def _fake_run_models(_img):
            # (results_ocr, results_table, paragraphs)
            return empty_ocr, tables, []

        monkeypatch.setattr(parser, "run_models", _fake_run_models)

        semantic_info, _, _ = parser(img, template=None, id=jf.stem)

        got = canonicalize_for_compare(dump_schema(semantic_info))
        golden_path = golden_dir / f"{jf.stem}.golden.json"

        # 初回のみゴールデン生成
        # golden_path.write_text(
        #    json.dumps(got, ensure_ascii=False, indent=2), encoding="utf-8"
        # )
        # pytest.fail(f"Golden not found. Created: {golden_path}. Re-run tests.")

        expected = json.loads(golden_path.read_text(encoding="utf-8"))
        assert_metrics_equal(got, expected)
