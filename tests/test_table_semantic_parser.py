# tests/test_table_semantic_contents_unit.py
from __future__ import annotations

import json
from pathlib import Path


from yomitoku.schemas.table_semantic_parser import (
    TableSemanticContentsSchema,
    CellSchema,
    KvItemSchema,
    TableGridSchema,
)

import yomitoku.schemas.table_semantic_parser as tsp_mod  # normalize/make_unique_all がここにある前提


# -------------------------
# helpers
# -------------------------
def mk_cell(
    cid: str, box, role="cell", contents="", row=0, col=0, row_span=1, col_span=1
):
    return CellSchema(
        id=cid,
        box=list(box),
        role=role,
        contents=contents,
        row=row,
        col=col,
        row_span=row_span,
        col_span=col_span,
        meta={},
    )


def mk_grid(grid_id: str, box, data, col_headers):
    return TableGridSchema(
        id=grid_id,
        box=list(box),
        n_row=len(data),
        n_col=len(data[0]) if data else 0,
        data=data,
        col_headers=col_headers,
    )


def mk_table(
    *,
    table_id="t0",
    style="border",
    box=(0, 0, 300, 200),
    cells=None,
    kv_items=None,
    grids=None,
):
    return TableSemanticContentsSchema(
        id=table_id,
        style=style,
        box=list(box),
        cells=cells or {},
        kv_items=kv_items or [],
        grids=grids or [],
    )


# -------------------------
# pure funcs: normalize / make_unique_all
# -------------------------
def test_normalize_removes_half_and_full_width_spaces():
    assert tsp_mod.normalize("a b　c 　d") == "abcd"


def test_make_unique_all_only_appends_index_for_duplicates():
    seq = [["A"], ["B"], ["A"], ["A"], ["C"]]
    out = tsp_mod.make_unique_all(seq)

    # B,C はユニークなのでそのまま
    assert ["B"] in out
    assert ["C"] in out

    # A は 3 個あるので idx 付き
    a_items = [x for x in out if x[0] == "A"]
    assert len(a_items) == 3
    assert sorted([x[1] for x in a_items]) == [0, 1, 2]


# -------------------------
# TableSemanticContentsSchema.safe_contents
# -------------------------
def test_safe_contents_returns_empty_when_cell_missing():
    t = mk_table(cells={})
    assert t.safe_contents("nope") == ""


def test_safe_contents_ignores_half_width_space_by_default():
    cells = {"c1": mk_cell("c1", (0, 0, 10, 10), contents="a b")}
    t = mk_table(cells=cells)
    assert t.safe_contents("c1") == "ab"


def test_safe_contents_can_keep_spaces_when_ignore_space_false():
    cells = {"c1": mk_cell("c1", (0, 0, 10, 10), contents="a b")}
    t = mk_table(cells=cells)
    assert t.safe_contents("c1", ignore_space=False) == "a b"


# -------------------------
# find_cell_by_id
# -------------------------
def test_find_cell_by_id_found_and_not_found():
    cells = {"c1": mk_cell("c1", (0, 0, 10, 10))}
    t = mk_table(cells=cells)

    assert t.find_cell_by_id("c1").id == "c1"
    assert t.find_cell_by_id("nope") is None


# -------------------------
# search_cells_by_bbox (uses is_contained)
# -------------------------
def test_search_cells_by_bbox_skips_group_role(monkeypatch):
    # is_contained を常に True にして「group が除外される」ことだけ確認
    monkeypatch.setattr(tsp_mod, "is_contained", lambda a, b, threshold=0.5: True)

    cells = {
        "g": mk_cell("g", (0, 0, 100, 100), role="group"),
        "c": mk_cell("c", (0, 0, 100, 100), role="cell"),
    }
    t = mk_table(cells=cells)

    out = t.search_cells_by_bbox([0, 0, 100, 100])
    assert [c.id for c in out] == ["c"]


def test_search_cells_by_bbox_returns_contained_cells(monkeypatch):
    # box が一致したら True の簡易 is_contained
    monkeypatch.setattr(
        tsp_mod,
        "is_contained",
        lambda q, c, threshold=0.5: list(q) == list(c),
    )

    cells = {
        "c1": mk_cell("c1", (0, 0, 10, 10)),
        "c2": mk_cell("c2", (10, 0, 20, 10)),
    }
    t = mk_table(cells=cells)

    out = t.search_cells_by_bbox([10, 0, 20, 10])
    assert [c.id for c in out] == ["c2"]


# -------------------------
# search_cells_by_query (normalize + role/group skip)
# -------------------------
def test_search_cells_by_query_matches_ignore_spaces_and_skips_group():
    cells = {
        "g": mk_cell("g", (0, 0, 10, 10), role="group", contents="契約番号"),
        "c1": mk_cell("c1", (0, 0, 10, 10), role="cell", contents="契約 番号"),
        "c2": mk_cell("c2", (0, 0, 10, 10), role="cell", contents="担当者"),
        "c3": mk_cell("c3", (0, 0, 10, 10), role="cell", contents=None),
    }
    t = mk_table(cells=cells)

    out = t.search_cells_by_query("契約番号")
    assert [c.id for c in out] == ["c1"]


# -------------------------
# relative search: below/right/left/upper
# (uses is_bottom_adjacent / is_right_adjacent)
# -------------------------
def test_search_cells_right_of_key_text(monkeypatch):
    # qセルは "Key" を含む
    key = mk_cell("k", (0, 0, 50, 50), contents="Key")
    right1 = mk_cell("r1", (50, 0, 100, 50), contents="V1")
    right2 = mk_cell("r2", (50, 50, 100, 100), contents="V2")
    other = mk_cell("o", (0, 50, 50, 100), contents="Other")

    cells = {c.id: c for c in [key, right1, right2, other]}
    t = mk_table(cells=cells)

    # 右隣判定: x1 == key.x2 かつ y-overlap
    def fake_is_right_adjacent(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        return bx1 == ax2 and not (by2 <= ay1 or ay2 <= by1)

    monkeypatch.setattr(tsp_mod, "is_right_adjacent", fake_is_right_adjacent)

    out = t.search_cells_right_of_key_text("Key")
    assert sorted([c.id for c in out]) == ["r1"]  # right2 は y が重ならないので除外


def test_search_cells_below_key_text(monkeypatch):
    key = mk_cell("k", (0, 0, 50, 50), contents="Key")
    below = mk_cell("b", (0, 50, 50, 100), contents="V")
    other = mk_cell("o", (50, 0, 100, 50), contents="Other")

    cells = {c.id: c for c in [key, below, other]}
    t = mk_table(cells=cells)

    def fake_is_bottom_adjacent(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        return by1 == ay2 and not (bx2 <= ax1 or ax2 <= bx1)

    monkeypatch.setattr(tsp_mod, "is_bottom_adjacent", fake_is_bottom_adjacent)

    out = t.search_cells_below_key_text("Key")
    assert [c.id for c in out] == ["b"]


def test_search_cells_left_of_key_text(monkeypatch):
    key = mk_cell("k", (50, 0, 100, 50), contents="Key")
    left = mk_cell("l", (0, 0, 50, 50), contents="L")
    other = mk_cell("o", (50, 50, 100, 100), contents="Other")

    cells = {c.id: c for c in [key, left, other]}
    t = mk_table(cells=cells)

    # left_of: is_right_adjacent(cell, query_cell) を使っているので注意
    def fake_is_right_adjacent(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        return bx1 == ax2 and not (by2 <= ay1 or ay2 <= by1)

    monkeypatch.setattr(tsp_mod, "is_right_adjacent", fake_is_right_adjacent)

    out = t.search_cells_left_of_key_text("Key")
    assert [c.id for c in out] == ["l"]


def test_search_cells_upper_key_text(monkeypatch):
    key = mk_cell("k", (0, 50, 50, 100), contents="Key")
    upper = mk_cell("u", (0, 0, 50, 50), contents="U")
    other = mk_cell("o", (50, 50, 100, 100), contents="Other")

    cells = {c.id: c for c in [key, upper, other]}
    t = mk_table(cells=cells)

    # upper: is_bottom_adjacent(cell, query_cell) を使っているので注意
    def fake_is_bottom_adjacent(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        return by1 == ay2 and not (bx2 <= ax1 or ax2 <= bx1)

    monkeypatch.setattr(tsp_mod, "is_bottom_adjacent", fake_is_bottom_adjacent)

    out = t.search_cells_upper_key_text("Key")
    assert [c.id for c in out] == ["u"]


# -------------------------
# TableSemanticContentsView.kv_items_to_dict
# -------------------------
def test_view_kv_items_to_dict_merges_keys_and_makes_unique():
    # kv_items: key=["k"], value="v"
    cells = {
        "k": mk_cell("k", (0, 0, 10, 10), role="header", contents="契約 番号"),
        "v": mk_cell("v", (10, 0, 20, 10), role="cell", contents=" 123 "),
        "k2": mk_cell("k2", (0, 10, 10, 20), role="header", contents="契約番号"),
        "v2": mk_cell("v2", (10, 10, 20, 20), role="cell", contents="456"),
    }
    kv_items = [
        KvItemSchema(id=None, key=["k"], value="v"),
        KvItemSchema(
            id=None, key=["k2"], value="v2"
        ),  # 同じキー文字列になる => make_unique_all で idx 付く
    ]
    t = mk_table(cells=cells, kv_items=kv_items)

    d = t.view.kv_items_to_dict()

    # normalize で "契約番号" に統一されるはず
    # make_unique_all の結果、片方は "['契約番号']_0" のように idx が入る（joinされる）
    assert len(d.keys()) == 2
    assert all("契約番号" in k for k in d.keys())
    assert set(d.values()) == {"123", "456"}  # safe_contents は半角space除去


# -------------------------
# TableSemanticContentsView.grids_to_dicts
# -------------------------
def test_view_grids_to_dicts_builds_row_dicts_and_skips_header_cells():
    # grid:
    # col_headers: [["h1"], ["h2"]]
    # data:
    #   ["h1", "h2"]  <- header行（cell が header id のため skipされる）
    #   ["a", "b"]
    cells = {
        "h1": mk_cell("h1", (0, 0, 10, 10), role="header", contents="項目"),
        "h2": mk_cell("h2", (10, 0, 20, 10), role="header", contents="値"),
        "a": mk_cell("a", (0, 10, 10, 20), role="cell", contents="AA"),
        "b": mk_cell("b", (10, 10, 20, 20), role="cell", contents="BB"),
    }
    grid = mk_grid(
        "g0",
        (0, 0, 20, 20),
        data=[["h1", "h2"], ["a", "b"]],
        col_headers=[["h1"], ["h2"]],
    )
    t = mk_table(cells=cells, grids=[grid])

    out = t.view.grids_to_dict()

    assert out == [
        {
            "id": "g0",
            "rows": [
                {"項目": "AA", "値": "BB"},
            ],
        }
    ]


def test_view_grids_to_dicts_avoids_duplicate_cell_id_in_same_row():
    # 1行で同じ cell_id が 2列に現れる場合は 2回目をスキップ
    cells = {
        "h1": mk_cell("h1", (0, 0, 10, 10), role="header", contents="A"),
        "h2": mk_cell("h2", (10, 0, 20, 10), role="header", contents="B"),
        "x": mk_cell("x", (0, 10, 10, 20), role="cell", contents="X"),
    }
    grid = mk_grid(
        "g0",
        (0, 0, 20, 20),
        data=[["x", "x"]],
        col_headers=[["h1"], ["h2"]],
    )
    t = mk_table(cells=cells, grids=[grid])

    out = t.view.grids_to_dict()
    # 片方だけ入る（どちらのキー側に残るかは実装順で決まるので集合で確認）
    row = out[0]["rows"][0]
    assert set(row.values()) == {"X"}
    assert len(row) == 1


# -------------------------
# TableSemanticContentsExport.to_json / grids_to_json / kv_items_to_json / grids_to_csv
# -------------------------
def test_export_to_json_writes_kv_and_grids(tmp_path: Path):
    cells = {
        "k": mk_cell("k", (0, 0, 10, 10), role="header", contents="Key"),
        "v": mk_cell("v", (10, 0, 20, 10), role="cell", contents="Val"),
        "h": mk_cell("h", (0, 0, 10, 10), role="header", contents="H"),
        "c": mk_cell("c", (0, 10, 10, 20), role="cell", contents="C"),
    }
    kv_items = [KvItemSchema(id=None, key=["k"], value="v")]
    grid = mk_grid("0", (0, 0, 20, 20), data=[["c"]], col_headers=[["h"]])
    t = mk_table(cells=cells, kv_items=kv_items, grids=[grid])

    out_path = tmp_path / "out" / "table.json"
    t.export.to_json(str(out_path))

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert "kv_items" in data
    assert "grids" in data
    assert isinstance(data["kv_items"], dict)
    assert isinstance(data["grids"], list)


def test_export_kv_items_to_json_writes_dict(tmp_path: Path):
    cells = {
        "k": mk_cell("k", (0, 0, 10, 10), role="header", contents="Key"),
        "v": mk_cell("v", (10, 0, 20, 10), role="cell", contents="Val"),
    }
    kv_items = [KvItemSchema(id=None, key=["k"], value="v")]
    t = mk_table(cells=cells, kv_items=kv_items, grids=[])

    out_dir = tmp_path / "kv"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "kv.json"

    t.export.kv_items_to_json(str(out_path))

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data == {"Key": "Val"}


def test_export_grids_to_json_writes_list(tmp_path: Path):
    cells = {
        "h": mk_cell("h", (0, 0, 10, 10), role="header", contents="H"),
        "c": mk_cell("c", (0, 10, 10, 20), role="cell", contents="C"),
    }
    grid = mk_grid("0", (0, 0, 20, 20), data=[["c"]], col_headers=[["h"]])
    t = mk_table(cells=cells, kv_items=[], grids=[grid])

    out_dir = tmp_path / "grids"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "grids.json"

    t.export.grids_to_json(str(out_path))

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert data[0]["id"] == "0"


def test_export_grids_to_csv_writes_csv_files(tmp_path: Path):
    cells = {
        "h1": mk_cell("h1", (0, 0, 10, 10), role="header", contents="A"),
        "h2": mk_cell("h2", (10, 0, 20, 10), role="header", contents="B"),
        "a": mk_cell("a", (0, 10, 10, 20), role="cell", contents="1 2"),
        "b": mk_cell("b", (10, 10, 20, 20), role="cell", contents="3"),
    }
    grid = mk_grid(
        "0",
        (0, 0, 20, 20),
        data=[["a", "b"]],
        col_headers=[["h1"], ["h2"]],
    )
    t = mk_table(cells=cells, kv_items=[], grids=[grid])

    out_base = tmp_path / "csv" / "out.csv"
    csvs = t.export.grids_to_csv(str(out_base), ignore_space=True)

    # 返り値（行列）
    assert csvs == [[["12", "3"]]]

    # 実ファイル: out_0.csv ができる
    out_file = tmp_path / "csv" / "out_0.csv"
    assert out_file.exists()
    assert out_file.read_text(encoding="utf-8").strip() == "12,3"
