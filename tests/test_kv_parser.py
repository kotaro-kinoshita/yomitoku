# tests/test_kv_parser.py

from types import SimpleNamespace

from yomitoku.kv_parser import parse_kv_items
from yomitoku.schemas.table_semantic_parser import RegionSchema


# -------------------------
# helpers
# -------------------------
def mk_cell(cid, box, role="cell", contents=""):
    return SimpleNamespace(
        id=cid,
        box=list(box),
        role=role,
        contents=contents,
        row=None,
        col=None,
        row_span=1,
        col_span=1,
    )


def mk_region(rid, box):
    return RegionSchema(id=rid, box=list(box), role="kv_item", score=1.0)


def build_nodes(headers=(), cells=(), empties=()):
    nodes = {"header": list(headers), "cell": list(cells), "empty": list(empties)}
    all_cells = {c.id: c for c in [*headers, *cells, *empties]}
    return nodes, all_cells


def kv_by_value(kv_items):
    return {kv.value: kv for kv in kv_items}


# -------------------------
# 孤児ヘッダーの入れ子救済
# -------------------------
def test_rescue_vertical_nested_orphan_header_chains_all_consecutive():
    """上下入れ子: 直下から下方向に連続するキーヘッダーすべてに紐付く"""
    # H (孤児・2列にまたがる上位ヘッダー)
    #  k1 | v1   <- region r0 (Hに直下隣接)
    #  k2 | v2   <- region r1 (k1に連続して隣接 -> Hに紐付く)
    #  (空白)
    #  k3 | v3   <- region r2 (連続が途切れている -> 紐付かない)
    h = mk_cell("H", (0, 0, 200, 20), role="header", contents="本店")
    k1 = mk_cell("k1", (0, 20, 60, 50), role="header", contents="フリガナ")
    v1 = mk_cell("v1", (60, 20, 200, 50), role="cell", contents="a")
    k2 = mk_cell("k2", (0, 50, 60, 80), role="header", contents="商号")
    v2 = mk_cell("v2", (60, 50, 200, 80), role="cell", contents="b")
    k3 = mk_cell("k3", (0, 150, 60, 180), role="header", contents="別欄")
    v3 = mk_cell("v3", (60, 150, 200, 180), role="cell", contents="c")

    nodes, cells = build_nodes(headers=[h, k1, k2, k3], cells=[v1, v2, v3])
    regions = [
        mk_region("r0", (0, 20, 200, 50)),
        mk_region("r1", (0, 50, 200, 80)),
        mk_region("r2", (0, 150, 200, 180)),
    ]

    kv_items, _, kv_cells = parse_kv_items(nodes, cells, regions)

    kvs = kv_by_value(kv_items)
    # 直下隣接のk1と、そこから連続して隣接するk2の両方にHが連結される
    assert kvs["v1"].key == ["H", "k1"]
    assert kvs["v2"].key == ["H", "k2"]
    # 連続が途切れたk3には連結されない
    assert kvs["v3"].key == ["k3"]
    assert "H" in kv_cells


def test_no_rescue_for_same_width_sibling_header():
    """同幅の直下ヘッダーは兄弟セクション見出しであり、入れ子扱いしない"""
    # H (孤児) の直下に同幅のキーヘッダー k1 (別セクションの行ヘッダー)
    h = mk_cell("H", (0, 0, 60, 50), role="header", contents="扶助状況")
    k1 = mk_cell("k1", (0, 50, 60, 100), role="header", contents="振込口座")
    v1 = mk_cell("v1", (60, 50, 200, 100), role="cell", contents="a")

    nodes, cells = build_nodes(headers=[h, k1], cells=[v1])
    regions = [mk_region("r0", (0, 50, 200, 100))]

    kv_items, _, _ = parse_kv_items(nodes, cells, regions)

    kvs = kv_by_value(kv_items)
    assert kvs["v1"].key == ["k1"]


def test_rescue_horizontal_nested_orphan_header():
    """左右入れ子: 孤児ヘッダーは右方の入れ子キーヘッダーすべてに紐付く"""
    # H (孤児・2行にまたがる行ヘッダー、領域とは非隣接)
    #   H | k1 | v1   <- region r0
    #   H | k2 | v2   <- region r1
    h = mk_cell("H", (0, 0, 30, 100), role="header", contents="連絡先")
    k1 = mk_cell("k1", (60, 0, 120, 45), role="header", contents="電話")
    v1 = mk_cell("v1", (120, 0, 250, 45), role="cell", contents="a")
    k2 = mk_cell("k2", (60, 55, 120, 100), role="header", contents="FAX")
    v2 = mk_cell("v2", (120, 55, 250, 100), role="cell", contents="b")

    nodes, cells = build_nodes(headers=[h, k1, k2], cells=[v1, v2])
    regions = [mk_region("r0", (60, 0, 250, 45)), mk_region("r1", (60, 55, 250, 100))]

    kv_items, _, kv_cells = parse_kv_items(nodes, cells, regions)

    kvs = kv_by_value(kv_items)
    # 右方の入れ子キーヘッダー「すべて」にHが連結される
    assert kvs["v1"].key == ["H", "k1"]
    assert kvs["v2"].key == ["H", "k2"]
    assert "H" in kv_cells


def test_no_rescue_for_non_nested_header():
    """入れ子でないヘッダー(x区間が重ならない等)には紐付かない"""
    h = mk_cell("H", (300, 0, 400, 20), role="header", contents="別の見出し")
    k1 = mk_cell("k1", (0, 20, 60, 50), role="header", contents="フリガナ")
    v1 = mk_cell("v1", (60, 20, 200, 50), role="cell", contents="a")

    nodes, cells = build_nodes(headers=[h, k1], cells=[v1])
    regions = [mk_region("r0", (0, 20, 200, 50))]

    kv_items, _, _ = parse_kv_items(nodes, cells, regions)

    kvs = kv_by_value(kv_items)
    assert kvs["v1"].key == ["k1"]


# -------------------------
# 孤児ヘッダー×孤児セルのKV化
# -------------------------
def test_rescue_orphan_header_with_right_adjacent_orphan_cell():
    """孤児ヘッダーと右隣接する孤児セルはKV化される"""
    h = mk_cell("H", (0, 0, 50, 30), role="header", contents="備考")
    c = mk_cell("C", (50, 0, 150, 30), role="cell", contents="特になし")

    nodes, cells = build_nodes(headers=[h], cells=[c])
    # 領域はどちらも含まない
    regions = [mk_region("r0", (300, 300, 400, 400))]

    kv_items, _, kv_cells = parse_kv_items(nodes, cells, regions)

    kvs = kv_by_value(kv_items)
    assert kvs["C"].key == ["H"]
    assert "H" in kv_cells


def test_rescue_orphan_header_with_bottom_adjacent_orphan_cell():
    """孤児ヘッダーと下隣接する孤児セルはKV化される"""
    h = mk_cell("H", (0, 0, 100, 30), role="header", contents="使用印鑑")
    c = mk_cell("C", (0, 30, 100, 120), role="cell", contents="")

    nodes, cells = build_nodes(headers=[h], cells=[c])
    regions = [mk_region("r0", (300, 300, 400, 400))]

    kv_items, _, _ = parse_kv_items(nodes, cells, regions)

    kvs = kv_by_value(kv_items)
    assert kvs["C"].key == ["H"]


def test_rescue_orphan_cell_chain_as_nested_key():
    """孤児セルの連鎖: 途中のセルはキーの一部、末端がvalueになる"""
    # H | c1 | c2  (すべて孤児、右方向の連鎖)
    # H | c3 | c4
    h = mk_cell("H", (0, 0, 50, 60), role="header", contents="扶助状況")
    c1 = mk_cell("c1", (50, 0, 150, 30), role="cell", contents="受給中の年金")
    c2 = mk_cell("c2", (150, 0, 300, 30), role="cell", contents="遺族年金")
    c3 = mk_cell("c3", (50, 30, 150, 60), role="cell", contents="援助·養育費等")
    c4 = mk_cell("c4", (150, 30, 300, 60), role="cell", contents="援助元")

    nodes, cells = build_nodes(headers=[h], cells=[c1, c2, c3, c4])
    regions = [mk_region("r0", (500, 500, 600, 600))]

    kv_items, _, kv_cells = parse_kv_items(nodes, cells, regions)

    kvs = kv_by_value(kv_items)
    # 連鎖の末端がvalue、途中のセルがキーに含まれる
    assert kvs["c2"].key == ["H", "c1"]
    assert kvs["c4"].key == ["H", "c3"]
    # キーとして使われたセルはキーなしkvとして重複出力されない
    assert "c1" not in kvs
    assert "c3" not in kvs
    assert "H" in kv_cells


def test_orphan_cell_without_adjacent_header_stays_keyless():
    """孤児ヘッダーと隣接しない孤児セルはキーなしのまま"""
    h = mk_cell("H", (0, 0, 50, 30), role="header", contents="備考")
    c = mk_cell("C", (500, 500, 600, 530), role="cell", contents="遠いセル")

    nodes, cells = build_nodes(headers=[h], cells=[c])
    regions = [mk_region("r0", (300, 0, 400, 100))]

    kv_items, _, _ = parse_kv_items(nodes, cells, regions)

    kvs = kv_by_value(kv_items)
    assert kvs["C"].key == []


# -------------------------
# 紐付き値セルの右方・下方に続く孤立セルへのキー引き継ぎ
# -------------------------
def test_orphan_cell_below_linked_value_inherits_key():
    """紐付き値セルの直下の孤立セルは同じキーを引き継ぐ (複数行の記入欄)"""
    # | k | v |   (領域)
    #     | c |   (孤児・vと同じ列幅)
    k = mk_cell("k", (0, 0, 60, 30), role="header", contents="申請理由")
    v = mk_cell("v", (60, 0, 300, 30), role="cell", contents="1行目")
    c = mk_cell("c", (60, 30, 300, 60), role="cell", contents="2行目")

    nodes, cells = build_nodes(headers=[k], cells=[v, c])
    regions = [mk_region("r0", (0, 0, 300, 30))]

    kv_items, _, kv_cells = parse_kv_items(nodes, cells, regions)

    kvs = kv_by_value(kv_items)
    assert kvs["v"].key == ["k"]
    assert kvs["c"].key == ["k"]
    assert "c" in kv_cells


def test_orphan_cell_right_of_linked_value_inherits_key():
    """紐付き値セルの右隣の孤立セルは同じキーを引き継ぎ、連鎖する"""
    # | k | v | c1 | c2 |  (c1, c2 は孤児・同じ行高)
    k = mk_cell("k", (0, 0, 60, 30), role="header", contents="項目")
    v = mk_cell("v", (60, 0, 120, 30), role="cell", contents="a")
    c1 = mk_cell("c1", (120, 0, 180, 30), role="cell", contents="b")
    c2 = mk_cell("c2", (180, 0, 240, 30), role="cell", contents="c")

    nodes, cells = build_nodes(headers=[k], cells=[v, c1, c2])
    regions = [mk_region("r0", (0, 0, 120, 30))]

    kv_items, _, _ = parse_kv_items(nodes, cells, regions)

    kvs = kv_by_value(kv_items)
    assert kvs["v"].key == ["k"]
    assert kvs["c1"].key == ["k"]
    # 孤立セル→孤立セルの連鎖でも引き継がれる
    assert kvs["c2"].key == ["k"]


def test_orphan_cell_wider_than_value_does_not_inherit():
    """値セルより大きくはみ出す孤立セル (全幅の注記など) は引き継がない"""
    # | k | v |          (領域)
    # |   注記セル    |   (孤児・kとvの全幅にまたがる)
    k = mk_cell("k", (0, 0, 100, 30), role="header", contents="メール")
    v = mk_cell("v", (100, 0, 300, 30), role="cell", contents="a@b.c")
    note = mk_cell("note", (0, 30, 300, 90), role="cell", contents="※注記")

    nodes, cells = build_nodes(headers=[k], cells=[v, note])
    regions = [mk_region("r0", (0, 0, 300, 30))]

    kv_items, _, _ = parse_kv_items(nodes, cells, regions)

    kvs = kv_by_value(kv_items)
    assert kvs["v"].key == ["k"]
    # 注記セルはキーなしのまま
    assert kvs["note"].key == []


def test_orphan_cell_next_to_keyless_value_stays_keyless():
    """キーなし値セルの隣の孤立セルは引き継がない (ヘッダー紐付きが前提)"""
    # | v | c |  (v は領域内だがキーを持たない)
    v = mk_cell("v", (0, 0, 60, 30), role="cell", contents="値")
    c = mk_cell("c", (60, 0, 120, 30), role="cell", contents="続き")

    nodes, cells = build_nodes(cells=[v, c])
    regions = [mk_region("r0", (0, 0, 60, 30))]

    kv_items, _, _ = parse_kv_items(nodes, cells, regions)

    kvs = kv_by_value(kv_items)
    assert kvs["v"].key == []
    assert kvs["c"].key == []
