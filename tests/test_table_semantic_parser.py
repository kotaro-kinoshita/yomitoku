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
def test_view_kv_items_to_dict_lists_same_text_sibling_keys():
    # 同じテキストを持つ別のキーセルは統合せず、値のリストになる
    cells = {
        "k": mk_cell("k", (0, 0, 10, 10), role="header", contents="契約 番号"),
        "v": mk_cell("v", (10, 0, 20, 10), role="cell", contents=" 123 "),
        "k2": mk_cell("k2", (0, 10, 10, 20), role="header", contents="契約番号"),
        "v2": mk_cell("v2", (10, 10, 20, 20), role="cell", contents="456"),
    }
    kv_items = [
        KvItemSchema(id=None, key=["k"], value="v"),
        KvItemSchema(id=None, key=["k2"], value="v2"),
    ]
    t = mk_table(cells=cells, kv_items=kv_items)

    d = t.view.kv_items_to_dict()

    # normalize で "契約番号" に統一され、同名キーの値がリストで並ぶ
    # (safe_contents は半角space除去)
    assert d == {"契約番号": ["123", "456"]}


def test_view_kv_items_to_dict_merge_vertical():
    """同一キーに対して縦方向に並ぶ複数のvalueを結合してソートする"""
    cells = {
        "k": mk_cell("k", (0, 0, 100, 30), role="header", contents="住所"),
        "v1": mk_cell("v1", (100, 0, 300, 30), role="cell", contents="東京都"),
        "v2": mk_cell("v2", (100, 30, 300, 60), role="cell", contents="新宿区"),
        "v3": mk_cell("v3", (100, 60, 300, 90), role="cell", contents="1-2-3"),
    }
    kv_items = [
        KvItemSchema(id=None, key=["k"], value="v1"),
        KvItemSchema(id=None, key=["k"], value="v3"),  # 意図的に順序を入れ替え
        KvItemSchema(id=None, key=["k"], value="v2"),
    ]
    t = mk_table(cells=cells, kv_items=kv_items)

    # 同一キーセルの複数valueはy座標でソートして結合される
    d = t.view.kv_items_to_dict()
    assert len(d) == 1
    assert d["住所"] == "東京都\n新宿区\n1-2-3"


def test_view_kv_items_to_dict_merge_horizontal():
    """同一キーに対して横方向に並ぶ複数のvalueを結合してソートする"""
    cells = {
        "k": mk_cell("k", (0, 0, 100, 50), role="header", contents="電話番号"),
        "v1": mk_cell("v1", (100, 0, 200, 50), role="cell", contents="03"),
        "v2": mk_cell("v2", (200, 0, 300, 50), role="cell", contents="1234"),
        "v3": mk_cell("v3", (300, 0, 400, 50), role="cell", contents="5678"),
    }
    kv_items = [
        KvItemSchema(id=None, key=["k"], value="v3"),  # 意図的に順序を入れ替え
        KvItemSchema(id=None, key=["k"], value="v1"),
        KvItemSchema(id=None, key=["k"], value="v2"),
    ]
    t = mk_table(cells=cells, kv_items=kv_items)

    # separator="-": x座標でソートして結合
    d = t.view.kv_items_to_dict(separator="-")
    assert len(d) == 1
    assert d["電話番号"] == "03-1234-5678"


def test_view_kv_items_to_dict_merge_single_value():
    """同一キーに1つのvalueしかない場合はそのまま返す"""
    cells = {
        "k": mk_cell("k", (0, 0, 100, 30), role="header", contents="名前"),
        "v": mk_cell("v", (100, 0, 300, 30), role="cell", contents="太郎"),
    }
    kv_items = [KvItemSchema(id=None, key=["k"], value="v")]
    t = mk_table(cells=cells, kv_items=kv_items)

    d = t.view.kv_items_to_dict()
    assert d == {"名前": "太郎"}


def test_view_kv_items_to_dict_merge_mixed_keys():
    """異なるキーが混在する場合、それぞれ個別にマージされる"""
    cells = {
        "k1": mk_cell("k1", (0, 0, 100, 30), role="header", contents="名前"),
        "k2": mk_cell("k2", (0, 30, 100, 60), role="header", contents="住所"),
        "v1": mk_cell("v1", (100, 0, 300, 30), role="cell", contents="太郎"),
        "v2": mk_cell("v2", (100, 30, 300, 60), role="cell", contents="東京都"),
        "v3": mk_cell("v3", (100, 60, 300, 90), role="cell", contents="新宿区"),
    }
    kv_items = [
        KvItemSchema(id=None, key=["k1"], value="v1"),
        KvItemSchema(id=None, key=["k2"], value="v2"),
        KvItemSchema(id=None, key=["k2"], value="v3"),
    ]
    t = mk_table(cells=cells, kv_items=kv_items)

    d = t.view.kv_items_to_dict()
    assert len(d) == 2
    assert d["名前"] == "太郎"
    assert d["住所"] == "東京都\n新宿区"


def test_view_kv_items_to_dict_nests_parent_headers():
    """親ヘッダーを持つキーは入れ子の階層dictになる"""
    cells = {
        "g": mk_cell("g", (0, 0, 300, 10), role="header", contents="申込者情報"),
        "k": mk_cell("k", (0, 10, 100, 40), role="header", contents="名前"),
        "v": mk_cell("v", (100, 10, 300, 40), role="cell", contents="太郎"),
    }
    kv_items = [KvItemSchema(id=None, key=["g", "k"], value="v")]
    t = mk_table(cells=cells, kv_items=kv_items)

    d = t.view.kv_items_to_dict()
    assert d == {"申込者情報": {"名前": "太郎"}}


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


# -------------------------
# TableSemanticContentsView.kv_items_to_structured / grids_to_structured
# -------------------------
def test_view_kv_items_to_structured_attaches_cells():
    cells = {
        "k": mk_cell("k", (0, 0, 100, 30), role="header", contents="名前"),
        "v": mk_cell("v", (100, 0, 300, 30), role="cell", contents="太郎"),
    }
    kv_items = [KvItemSchema(id="kv0", key=["k"], value="v")]
    t = mk_table(cells=cells, kv_items=kv_items)

    entries = t.view.kv_items_to_structured()

    assert len(entries) == 1
    e = entries[0]
    assert e.key == ["名前"]
    assert e.value == "太郎"
    assert [c.id for c in e.key_cells] == ["k"]
    assert [list(c.box) for c in e.key_cells] == [[0, 0, 100, 30]]
    assert [c.id for c in e.value_cells] == ["v"]
    assert [list(c.box) for c in e.value_cells] == [[100, 0, 300, 30]]


def test_view_kv_items_to_structured_merges_same_key_cell_vertically():
    """同一キーセルの複数valueは空間順に結合され、value_cellsが順序を保持する"""
    cells = {
        "k": mk_cell("k", (0, 0, 100, 60), role="header", contents="住所"),
        "v1": mk_cell("v1", (100, 0, 300, 30), role="cell", contents="東京都"),
        "v2": mk_cell("v2", (100, 30, 300, 60), role="cell", contents="新宿区"),
    }
    kv_items = [
        KvItemSchema(id=None, key=["k"], value="v2"),  # 意図的に逆順
        KvItemSchema(id=None, key=["k"], value="v1"),
    ]
    t = mk_table(cells=cells, kv_items=kv_items)

    entries = t.view.kv_items_to_structured()

    assert len(entries) == 1
    e = entries[0]
    assert e.key == ["住所"]
    assert e.value == "東京都\n新宿区"  # y座標でソートされる
    assert [c.id for c in e.value_cells] == ["v1", "v2"]
    assert [c.id for c in e.key_cells] == ["k"]


def test_view_kv_items_to_structured_keeps_same_text_different_cells_separate():
    """同じラベル文字列でも別のキーセルなら統合しない"""
    cells = {
        "k1": mk_cell("k1", (0, 0, 100, 30), role="header", contents="住所"),
        "k2": mk_cell("k2", (0, 200, 100, 230), role="header", contents="住所"),
        "v1": mk_cell("v1", (100, 0, 300, 30), role="cell", contents="東京都"),
        "v2": mk_cell("v2", (100, 200, 300, 230), role="cell", contents="大阪府"),
    }
    kv_items = [
        KvItemSchema(id=None, key=["k1"], value="v1"),
        KvItemSchema(id=None, key=["k2"], value="v2"),
    ]
    t = mk_table(cells=cells, kv_items=kv_items)

    entries = t.view.kv_items_to_structured()

    assert len(entries) == 2
    assert [(e.key, e.value) for e in entries] == [
        (["住所"], "東京都"),
        (["住所"], "大阪府"),
    ]
    assert [c.id for c in entries[0].key_cells] == ["k1"]
    assert [c.id for c in entries[1].key_cells] == ["k2"]


def test_view_kv_items_to_structured_normalizes_bare_str_key():
    """key が bare str でも文字単位に分解されない"""
    cells = {
        "k": mk_cell("k", (0, 0, 100, 30), role="header", contents="名前"),
        "v": mk_cell("v", (100, 0, 300, 30), role="cell", contents="太郎"),
    }
    kv_items = [KvItemSchema(id=None, key="k", value="v")]
    t = mk_table(cells=cells, kv_items=kv_items)

    entries = t.view.kv_items_to_structured()

    assert len(entries) == 1
    assert entries[0].key == ["名前"]
    assert [c.id for c in entries[0].key_cells] == ["k"]


def test_view_kv_items_to_structured_skips_missing_cells():
    """cells に存在しないIDはテキスト空・セル参照なしで安全に処理される"""
    cells = {
        "k": mk_cell("k", (0, 0, 100, 30), role="header", contents="名前"),
    }
    kv_items = [KvItemSchema(id=None, key=["k"], value="missing")]
    t = mk_table(cells=cells, kv_items=kv_items)

    entries = t.view.kv_items_to_structured()

    assert len(entries) == 1
    assert entries[0].value == ""
    assert entries[0].value_cells == []


def test_view_grids_to_structured_resolves_headers_and_cells():
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

    out = t.view.grids_to_structured()

    assert len(out) == 1
    g = out[0]
    assert g.id == "g0"
    assert g.n_row == 2 and g.n_col == 2
    # ヘッダ行は除外され、データ行のみ
    assert len(g.rows) == 1
    row = g.rows[0].cells
    assert [(e.key, e.value) for e in row] == [(["項目"], "AA"), (["値"], "BB")]
    assert [c.id for c in row[0].key_cells] == ["h1"]
    assert [c.id for c in row[0].value_cells] == ["a"]
    assert [list(c.box) for c in row[0].value_cells] == [[0, 10, 10, 20]]


def test_view_grids_to_structured_handles_none_holes_and_long_rows():
    """data の None 穴はスキップ、col_headers より長い行は安全に打ち切る"""
    cells = {
        "h1": mk_cell("h1", (0, 0, 10, 10), role="header", contents="A"),
        "a": mk_cell("a", (0, 10, 10, 20), role="cell", contents="X"),
        "b": mk_cell("b", (10, 10, 20, 20), role="cell", contents="Y"),
    }
    grid = mk_grid(
        "g0",
        (0, 0, 20, 20),
        data=[[None, "a", "b"]],  # col_headers は1列分のみ
        col_headers=[["h1"]],
    )
    t = mk_table(cells=cells, grids=[grid])

    out = t.view.grids_to_structured()

    # None はスキップ、範囲外の "a"(idx1)/"b"(idx2) は打ち切りで空行 -> rows なし
    assert out[0].rows == []


def test_to_structured_builds_document_with_paragraphs():
    from yomitoku.schemas import Element
    from yomitoku.schemas.table_semantic_parser import TableSemanticParserSchema

    cells = {
        "k": mk_cell("k", (0, 0, 100, 30), role="header", contents="名前"),
        "v": mk_cell("v", (100, 0, 300, 30), role="cell", contents="太郎"),
    }
    kv_items = [KvItemSchema(id="kv0", key=["k"], value="v")]
    table = mk_table(cells=cells, kv_items=kv_items)
    paragraph = Element(
        id="p0", box=[0, 100, 200, 130], score=0.9, role=None, contents="本文"
    )
    doc = TableSemanticParserSchema(tables=[table], paragraphs=[paragraph], words=[])

    structured = doc.to_structured()

    assert len(structured.tables) == 1
    assert structured.tables[0].id == "t0"
    assert structured.tables[0].kv_items[0].key == ["名前"]
    assert len(structured.paragraphs) == 1
    assert structured.paragraphs[0].contents == "本文"

    # BaseSchema 経由で JSON シリアライズ可能
    dumped = structured.model_dump()
    assert dumped["tables"][0]["kv_items"][0]["value_cells"][0]["id"] == "v"


def test_to_simple_drops_metadata_and_keeps_texts():
    from yomitoku.schemas import Element
    from yomitoku.schemas.table_semantic_parser import TableSemanticParserSchema

    cells = {
        "k": mk_cell("k", (0, 0, 100, 60), role="header", contents="住所"),
        "k2": mk_cell("k2", (0, 200, 100, 230), role="header", contents="住所"),
        "v1": mk_cell("v1", (100, 0, 300, 30), role="cell", contents="東京都"),
        "v2": mk_cell("v2", (100, 30, 300, 60), role="cell", contents="新宿区"),
        "v3": mk_cell("v3", (100, 200, 300, 230), role="cell", contents="大阪府"),
        "h1": mk_cell("h1", (0, 100, 10, 110), role="header", contents="項目"),
        "h2": mk_cell("h2", (10, 100, 20, 110), role="header", contents="値"),
        "a": mk_cell("a", (0, 110, 10, 120), role="cell", contents="AA"),
        "b": mk_cell("b", (10, 110, 20, 120), role="cell", contents="BB"),
    }
    kv_items = [
        # 同一キーセル k に2つの値 -> 結合される
        KvItemSchema(id=None, key=["k"], value="v1"),
        KvItemSchema(id=None, key=["k"], value="v2"),
        # 同名テキストだが別セル k2 -> 結合されず _0/_1 で区別される
        KvItemSchema(id=None, key=["k2"], value="v3"),
    ]
    grid = mk_grid(
        "g0",
        (0, 100, 20, 120),
        data=[["h1", "h2"], ["a", "b"]],
        col_headers=[["h1"], ["h2"]],
    )
    table = mk_table(cells=cells, kv_items=kv_items, grids=[grid])
    paragraph = Element(
        id="p0", box=[0, 200, 200, 230], score=0.9, role=None, contents="本文"
    )
    doc = TableSemanticParserSchema(tables=[table], paragraphs=[paragraph], words=[])

    simple = doc.to_simple()

    assert len(simple.tables) == 1
    t = simple.tables[0]
    # 同一キーセルは結合、同名別セルは配列で区別される
    assert t.kv_items == {"住所": ["東京都\n新宿区", "大阪府"]}
    assert t.grids[0].id == "g0"
    assert t.grids[0].rows == [{"項目": "AA", "値": "BB"}]
    assert simple.paragraphs == ["本文"]

    # 座標・セル参照が一切含まれない
    dumped = simple.model_dump()
    assert "box" not in json.dumps(dumped, ensure_ascii=False)


def test_to_simple_suffixes_duplicate_headers_in_grid_row():
    from yomitoku.schemas.table_semantic_parser import TableSemanticParserSchema

    cells = {
        "h1": mk_cell("h1", (0, 0, 10, 10), role="header", contents="日付"),
        "h2": mk_cell("h2", (10, 0, 20, 10), role="header", contents="日付"),
        "a": mk_cell("a", (0, 10, 10, 20), role="cell", contents="1月"),
        "b": mk_cell("b", (10, 10, 20, 20), role="cell", contents="2月"),
    }
    grid = mk_grid(
        "g0",
        (0, 0, 20, 20),
        data=[["a", "b"]],
        col_headers=[["h1"], ["h2"]],
    )
    table = mk_table(cells=cells, grids=[grid])
    doc = TableSemanticParserSchema(tables=[table], paragraphs=[], words=[])

    simple = doc.to_simple()

    # 行内でヘッダテキストが重複する場合は _0/_1 で区別され、値が失われない
    assert simple.tables[0].grids[0].rows == [{"日付_0": "1月", "日付_1": "2月"}]


def test_view_kv_items_to_structured_does_not_merge_keyless_cells():
    """キーなしの単独セルは同一(空)キーでも結合されない"""
    cells = {
        "v1": mk_cell("v1", (0, 0, 100, 30), role="cell", contents="I_SUPER流通系"),
        "v2": mk_cell("v2", (0, 30, 100, 60), role="cell", contents="請求"),
    }
    kv_items = [
        KvItemSchema(id=None, key=[], value="v1"),
        KvItemSchema(id=None, key=[], value="v2"),
    ]
    t = mk_table(cells=cells, kv_items=kv_items)

    entries = t.view.kv_items_to_structured()

    assert len(entries) == 2
    assert [(e.key, e.value) for e in entries] == [
        ([], "I_SUPER流通系"),
        ([], "請求"),
    ]
    assert [c.id for c in entries[0].value_cells] == ["v1"]
    assert [c.id for c in entries[1].value_cells] == ["v2"]


def test_to_simple_keeps_keyless_cells_separate_as_list():
    """--simple でもキーなしセルは結合されず、_unkeyed の下に配列で並ぶ"""
    from yomitoku.schemas.table_semantic_parser import TableSemanticParserSchema

    cells = {
        "v1": mk_cell("v1", (0, 0, 100, 30), role="cell", contents="A"),
        "v2": mk_cell("v2", (0, 30, 100, 60), role="cell", contents="B"),
    }
    kv_items = [
        KvItemSchema(id=None, key=[], value="v1"),
        KvItemSchema(id=None, key=[], value="v2"),
    ]
    table = mk_table(cells=cells, kv_items=kv_items)
    doc = TableSemanticParserSchema(tables=[table], paragraphs=[], words=[])

    simple = doc.to_simple()

    assert simple.tables[0].kv_items == {"_unkeyed": ["A", "B"]}


def test_view_kv_items_to_nested_groups_by_parent_header():
    """入れ子ヘッダーは親ヘッダーでグループ化された階層dictになる"""
    cells = {
        "p": mk_cell("p", (0, 0, 20, 60), role="header", contents="申込者情報"),
        "k1": mk_cell("k1", (20, 0, 60, 30), role="header", contents="団体名"),
        "k2": mk_cell("k2", (20, 30, 60, 60), role="header", contents="電話番号"),
        "v1": mk_cell("v1", (60, 0, 200, 30), role="cell", contents="MLism"),
        "v2": mk_cell("v2", (60, 30, 200, 60), role="cell", contents="090"),
        "s": mk_cell("s", (0, 60, 60, 90), role="header", contents="備考"),
        "sv": mk_cell("sv", (60, 60, 200, 90), role="cell", contents="なし"),
    }
    kv_items = [
        KvItemSchema(id=None, key=["p", "k1"], value="v1"),
        KvItemSchema(id=None, key=["p", "k2"], value="v2"),
        KvItemSchema(id=None, key=["s"], value="sv"),
    ]
    t = mk_table(cells=cells, kv_items=kv_items)

    nested = t.view.kv_items_to_nested()

    assert nested == {
        "申込者情報": {"団体名": "MLism", "電話番号": "090"},
        "備考": "なし",
    }


def test_view_kv_items_to_nested_repeated_blocks_become_list():
    """同名テキストの別親ヘッダー (繰り返しブロック) は配列になる"""
    cells = {
        "p1": mk_cell("p1", (0, 0, 20, 60), role="header", contents="扶養親族"),
        "p2": mk_cell("p2", (0, 60, 20, 120), role="header", contents="扶養親族"),
        "n1": mk_cell("n1", (20, 0, 60, 30), role="header", contents="氏名"),
        "n2": mk_cell("n2", (20, 60, 60, 90), role="header", contents="氏名"),
        "v1": mk_cell("v1", (60, 0, 200, 30), role="cell", contents="山田太郎"),
        "v2": mk_cell("v2", (60, 60, 200, 90), role="cell", contents="山田花子"),
    }
    kv_items = [
        KvItemSchema(id=None, key=["p1", "n1"], value="v1"),
        KvItemSchema(id=None, key=["p2", "n2"], value="v2"),
    ]
    t = mk_table(cells=cells, kv_items=kv_items)

    nested = t.view.kv_items_to_nested()

    assert nested == {
        "扶養親族": [{"氏名": "山田太郎"}, {"氏名": "山田花子"}],
    }


def test_view_kv_items_to_nested_parent_with_value_and_children():
    """親キーが値と子キーの両方を持つ場合、値は _value に入る"""
    cells = {
        "p": mk_cell("p", (0, 0, 20, 60), role="header", contents="扶養親族"),
        "pv": mk_cell("pv", (20, 0, 60, 30), role="cell", contents="有"),
        "n": mk_cell("n", (20, 30, 60, 60), role="header", contents="氏名"),
        "nv": mk_cell("nv", (60, 30, 200, 60), role="cell", contents="山田太郎"),
    }
    kv_items = [
        KvItemSchema(id=None, key=["p"], value="pv"),
        KvItemSchema(id=None, key=["p", "n"], value="nv"),
    ]
    t = mk_table(cells=cells, kv_items=kv_items)

    nested = t.view.kv_items_to_nested()

    assert nested == {
        "扶養親族": {"_value": "有", "氏名": "山田太郎"},
    }


def test_view_kv_items_to_nested_merges_values_on_same_key_chain():
    """同一キーセル列の複数 value は to_structured と同様に separator 結合"""
    cells = {
        "p": mk_cell("p", (0, 0, 20, 60), role="header", contents="住所"),
        "v1": mk_cell("v1", (20, 0, 200, 30), role="cell", contents="東京都"),
        "v2": mk_cell("v2", (20, 30, 200, 60), role="cell", contents="新宿区"),
    }
    kv_items = [
        KvItemSchema(id=None, key=["p"], value="v1"),
        KvItemSchema(id=None, key=["p"], value="v2"),
    ]
    t = mk_table(cells=cells, kv_items=kv_items)

    nested = t.view.kv_items_to_nested(separator="-")

    assert nested == {"住所": "東京都-新宿区"}


def test_resolve_overlapping_grid_regions_keeps_higher_score():
    """重複するgrid領域予測は検出スコアが高い方だけ残る"""
    from yomitoku.table_semantic_parser import _resolve_overlapping_regions
    from yomitoku.schemas.table_semantic_parser import RegionSchema

    cells = [
        mk_cell(f"c{r}{c}", (c * 100, r * 50, (c + 1) * 100, (r + 1) * 50))
        for r in range(3)
        for c in range(3)
    ]
    # 小さい方 (2列分) がスコア高、大きい方 (3列分) がスコア低
    small = RegionSchema(id="g_small", box=[0, 0, 200, 150], role="grid", score=0.99)
    large = RegionSchema(id="g_large", box=[0, 0, 300, 150], role="grid", score=0.98)

    grids, kvs = _resolve_overlapping_regions([large, small], [], cells)

    assert len(grids) == 1
    assert grids[0].id == "g_small"


def test_resolve_non_overlapping_grid_regions_all_kept():
    """重複しないgrid領域はすべて残る"""
    from yomitoku.table_semantic_parser import _resolve_overlapping_regions
    from yomitoku.schemas.table_semantic_parser import RegionSchema

    cells = [
        mk_cell("a", (0, 0, 100, 50)),
        mk_cell("b", (0, 500, 100, 550)),
    ]
    g1 = RegionSchema(id="g1", box=[0, 0, 100, 50], role="grid", score=0.9)
    g2 = RegionSchema(id="g2", box=[0, 500, 100, 550], role="grid", score=0.9)

    grids, _ = _resolve_overlapping_regions([g1, g2], [], cells)

    assert len(grids) == 2


def test_cell_id_visualizer_draws_chips_and_skips_group():
    import numpy as np

    from yomitoku.cli.table import FONT_PATH
    from yomitoku.utils.visualizer import cell_id_visualizer

    cells = {
        "c0": mk_cell("c0", (50, 50, 150, 100), role="header", contents="k"),
        "g0": mk_cell("g0", (200, 150, 290, 190), role="group"),
    }
    t = mk_table(cells=cells)
    img = np.full((200, 300, 3), 255, dtype=np.uint8)

    out = cell_id_visualizer(img, [t], FONT_PATH, font_size=14)

    assert out.shape == img.shape
    assert (img == 255).all()  # 入力は破壊されない
    # c0 の左上にはチップが描かれる
    assert (out[52:70, 52:70] != 255).any()
    # group セル (g0) の左上には描かれない
    assert (out[152:170, 202:220] == 255).all()
