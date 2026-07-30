import json
from pathlib import Path

import pytest

from yomitoku.cli import table


# -------------------------
# 引数バリデーション (モデルロードなし)
# -------------------------
def test_run_not_exist(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        [
            "table.py",
            "tests/data/dummy",
            "-o",
            str(tmp_path),
        ],
    )
    with pytest.raises(FileNotFoundError):
        table.main()


def test_run_not_exist_template(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        [
            "table.py",
            "tests/data/test.jpg",
            "-o",
            str(tmp_path),
            "--template",
            "tests/data/dummy_template.json",
        ],
    )
    with pytest.raises(FileNotFoundError):
        table.main()


def test_run_raw_and_simple_conflict(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        [
            "table.py",
            "tests/data/test.jpg",
            "-o",
            str(tmp_path),
            "--raw",
            "--simple",
        ],
    )
    with pytest.raises(ValueError):
        table.main()


def test_run_invalid_encoding(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        [
            "table.py",
            "tests/data/test.jpg",
            "-o",
            str(tmp_path),
            "--encoding",
            "invalid",
        ],
    )
    with pytest.raises(ValueError):
        table.main()


# -------------------------
# E2E (実モデルでの実行)
# -------------------------
def test_run_jpg_structured_with_vis(monkeypatch, tmp_path):
    """デフォルト(構造化JSON) + 可視化画像の出力"""
    monkeypatch.setattr(
        "sys.argv",
        [
            "table.py",
            "demo/table.jpg",
            "-o",
            str(tmp_path),
            "-v",
        ],
    )
    table.main()

    out_path = tmp_path / "table_p1.json"
    assert out_path.exists()
    assert (tmp_path / "table_p1_layout.jpg").exists()
    assert (tmp_path / "table_p1_ocr.jpg").exists()

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert isinstance(data["tables"], list)
    assert isinstance(data["paragraphs"], list)
    assert len(data["tables"]) > 0

    t0 = data["tables"][0]
    assert set(t0.keys()) == {"id", "box", "style", "kv_items", "grids"}
    assert len(t0["kv_items"]) > 0

    # 解決済みエントリの形: テキストと由来セル(ID+座標)を持つ
    kv = t0["kv_items"][0]
    assert set(kv.keys()) == {"key", "value", "key_cells", "value_cells"}
    assert len(kv["value_cells"]) >= 1
    ref = kv["value_cells"][0]
    assert set(ref.keys()) == {"id", "box"}
    assert len(ref["box"]) == 4

    # paragraphs はテキストと座標を持つ
    p0 = data["paragraphs"][0]
    assert {"id", "box", "contents"} <= set(p0.keys())


def test_run_jpg_simple(monkeypatch, tmp_path):
    """--simple: 座標・セル参照を含まないテキストのみの出力"""
    monkeypatch.setattr(
        "sys.argv",
        [
            "table.py",
            "demo/table.jpg",
            "-o",
            str(tmp_path),
            "--simple",
        ],
    )
    table.main()

    out_path = tmp_path / "table_p1.json"
    data = json.loads(out_path.read_text(encoding="utf-8"))

    t0 = data["tables"][0]
    assert isinstance(t0["kv_items"], dict)
    # 値は文字列・階層dict・繰り返しブロックの配列のいずれか
    assert all(isinstance(v, (str, dict, list)) for v in t0["kv_items"].values())
    assert all(isinstance(p, (str, type(None))) for p in data["paragraphs"])

    # メタ情報 (座標・セル参照) が含まれない
    dumped = json.dumps(data, ensure_ascii=False)
    assert '"box"' not in dumped
    assert "key_cells" not in dumped


def test_run_jpg_raw(monkeypatch, tmp_path):
    """--raw: 正規化スキーマ (cells/words を含むロスレス形式)"""
    monkeypatch.setattr(
        "sys.argv",
        [
            "table.py",
            "demo/table.jpg",
            "-o",
            str(tmp_path),
            "--raw",
        ],
    )
    table.main()

    out_path = tmp_path / "table_p1.json"
    data = json.loads(out_path.read_text(encoding="utf-8"))

    assert set(data.keys()) == {"tables", "paragraphs", "words"}
    t0 = data["tables"][0]
    # 正規化スキーマ: cells辞書を持ち、kv_itemsはセルID参照
    assert isinstance(t0["cells"], dict)
    assert len(t0["kv_items"]) > 0
    kv = t0["kv_items"][0]
    assert kv["value"] in t0["cells"]


def test_run_pdf_with_pages(monkeypatch, tmp_path):
    """PDF入力: --pages によるページ指定"""
    monkeypatch.setattr(
        "sys.argv",
        [
            "table.py",
            "tests/data/test.pdf",
            "-o",
            str(tmp_path),
            "--pages",
            "1",
        ],
    )
    table.main()

    outputs = sorted(p.name for p in Path(tmp_path).glob("*.json"))
    assert outputs == ["test_p1.json"]

    data = json.loads((tmp_path / "test_p1.json").read_text(encoding="utf-8"))
    assert "tables" in data and "paragraphs" in data
