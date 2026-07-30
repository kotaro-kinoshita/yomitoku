# tests/test_table_semantic_parser_utils.py

from types import SimpleNamespace
import networkx as nx

# 例: from yomitoku.table_semantic_parser import ...
from yomitoku.table_semantic_parser import (
    _split_nodes_with_role,
    get_cell_by_id,
    _get_cluster_nodes,
    drop_single_out_edge_by_type,
    replace_edge_type,
    sort_cells,
    _sort_elements,
    _assign_ids,
)


# -------------------------
# helpers
# -------------------------
def mk_cell(id, box, role="cell", contents=""):
    # TableSemanticParser 側で参照される属性だけ持てばOK
    return SimpleNamespace(
        id=id,
        box=box,
        role=role,
        contents=contents,
        row=None,
        col=None,
        row_span=1,
        col_span=1,
    )


def mk_elem(id, box):
    return SimpleNamespace(id=id, box=box)


# ============================================================
# _split_nodes_with_role / get_cell_by_id / _get_cluster_nodes
# ============================================================


def test_split_nodes_with_role_basic():
    cells = [
        mk_cell("h0", (0, 0, 10, 10), role="header"),
        mk_cell("c0", (0, 10, 10, 20), role="cell"),
        mk_cell("e0", (10, 10, 20, 20), role="empty"),
        mk_cell("g0", (0, 20, 20, 30), role="group"),
        mk_cell("x0", (0, 30, 20, 40), role="weird_role"),
    ]
    nodes = _split_nodes_with_role(cells)

    assert set(nodes.keys()) >= {"header", "cell", "empty", "group"}
    assert [c.id for c in nodes["header"]] == ["h0"]
    assert [c.id for c in nodes["cell"]] == ["c0"]
    assert [c.id for c in nodes["empty"]] == ["e0"]
    assert [c.id for c in nodes["group"]] == ["g0"]
    assert [c.id for c in nodes["weird_role"]] == ["x0"]

    nodes = _split_nodes_with_role([])
    assert nodes["header"] == []
    assert nodes["cell"] == []
    assert nodes["empty"] == []
    # "group" はプリセットしない（kv グラフは領域内セルから直接構築する）
    assert "group" not in nodes


def test_get_cell_by_id_found_and_not_found():
    cells = [mk_cell("a", (0, 0, 1, 1)), mk_cell("b", (0, 0, 1, 1))]
    assert get_cell_by_id(cells, "b").id == "b"
    assert get_cell_by_id(cells, "zzz") is None


def test_get_cluster_nodes_picks_correct_roles():
    # nodes は role ごとに list
    nodes = {
        "header": [mk_cell("h0", (0, 0, 10, 10), role="header")],
        "cell": [mk_cell("c0", (0, 10, 10, 20), role="cell")],
        "empty": [mk_cell("e0", (10, 10, 20, 20), role="empty")],
        "group": [mk_cell("g0", (0, 20, 20, 30), role="group")],
    }
    clusters = [set(["h0", "c0", "e0"])]
    out = _get_cluster_nodes(clusters, nodes)

    assert len(out) == 1
    clustered = out[0]
    assert [c.id for c in clustered["header"]] == ["h0"]
    assert [c.id for c in clustered["cell"]] == ["c0"]
    assert [c.id for c in clustered["empty"]] == ["e0"]
    # group は対象外（get_cell_by_id の検索対象が header/cell/empty なので入らない）
    assert "group" not in clustered or clustered["group"] == []

    out = _get_cluster_nodes([], nodes)
    assert out == []


# ============================================================
# drop_single_out_edge_by_type / replace_edge_type
# ============================================================


def test_drop_single_out_edge_by_type_removes_only_singletons():
    G = nx.DiGraph()
    G.add_nodes_from(["a", "b", "c", "d", "e"])
    # a -> b が type="X" で1本だけ -> 消える
    G.add_edge("a", "b", type="X")
    # c -> d, c -> e が type="X" で2本 -> 消えない
    G.add_edge("c", "d", type="X")
    G.add_edge("c", "e", type="X")
    # a -> c が type="Y" -> 対象外で残る
    G.add_edge("a", "c", type="Y")

    removed = drop_single_out_edge_by_type(G, edge_type="X", type_key="type")
    assert ("a", "b") in removed
    assert ("c", "d") not in removed
    assert ("c", "e") not in removed

    assert not G.has_edge("a", "b")
    assert G.has_edge("c", "d")
    assert G.has_edge("c", "e")
    assert G.has_edge("a", "c")


def test_replace_edge_type_rewrites_matching_only():
    G = nx.DiGraph()
    G.add_edge("a", "b", type="X")
    G.add_edge("b", "c", type="Y")
    G.add_edge("c", "d", type="X")

    replace_edge_type(G, from_type="X", to_type="Z", key="type")

    assert G["a"]["b"]["type"] == "Z"
    assert G["b"]["c"]["type"] == "Y"
    assert G["c"]["d"]["type"] == "Z"


# ============================================================
# sort_cells / _sort_elements / _assign_ids
# ============================================================


def test_sort_cells_assigns_position_ids_and_orders_values_before_groups():
    # min_height=10 として並びが安定するように作る
    c0 = mk_cell("old0", (0, 0, 10, 10), role="cell")
    c1 = mk_cell("old1", (20, 0, 30, 10), role="header")
    c2 = mk_cell("old2", (0, 20, 10, 30), role="empty")
    g0 = mk_cell("grp", (0, 5, 50, 60), role="group")

    cells, remap = sort_cells([g0, c2, c0, c1])

    # values(cell/header/empty) が先、groups が後
    assert [c.role for c in cells[:-1]] == ["cell", "header", "empty"]
    assert cells[-1].role == "group"

    # 位置ベースの r{行}c{列} が付く (group は grp 連番)
    assert set(remap.keys()) == {"old0", "old1", "old2", "grp"}
    assert [c.id for c in cells] == ["r0c0", "r0c1", "r1c0", "grp0"]

    cells, remap = sort_cells([])
    assert cells == []
    assert remap == {}


def test_sort_cells_position_ids_ignore_other_cell_count_changes():
    """セルが1つ増えても、既存セルの ID は変わらない (連番との違い)"""
    base = [
        mk_cell("a", (0, 0, 10, 10), role="cell"),
        mk_cell("b", (100, 0, 110, 10), role="cell"),
        mk_cell("c", (0, 100, 10, 110), role="cell"),
    ]
    _, remap1 = sort_cells([mk_cell(c.id, c.box, role=c.role) for c in base])

    extra = base + [mk_cell("x", (100, 100, 110, 110), role="cell")]
    _, remap2 = sort_cells([mk_cell(c.id, c.box, role=c.role) for c in extra])

    for old in ["a", "b", "c"]:
        assert remap1[old] == remap2[old]


def test_sort_cells_position_ids_stable_under_jitter():
    """数pxの座標ジッタでは ID が変わらない"""
    cells = [
        mk_cell("a", (0, 0, 100, 30), role="cell"),
        mk_cell("b", (100, 2, 200, 32), role="cell"),
        mk_cell("c", (1, 60, 101, 90), role="cell"),
        mk_cell("d", (102, 61, 202, 91), role="cell"),
    ]
    _, remap1 = sort_cells([mk_cell(c.id, c.box, role=c.role) for c in cells])

    jittered = [
        mk_cell("a", (3, 2, 103, 32), role="cell"),
        mk_cell("b", (97, 0, 197, 30), role="cell"),
        mk_cell("c", (0, 63, 100, 93), role="cell"),
        mk_cell("d", (105, 58, 205, 88), role="cell"),
    ]
    _, remap2 = sort_cells(jittered)

    assert remap1 == remap2
    assert remap1 == {"a": "r0c0", "b": "r0c1", "c": "r1c0", "d": "r1c1"}


def test_sort_cells_same_row_cells_get_distinct_ordinals():
    """同一行のほぼ同位置のセルも行内序数で区別され、値が失われない"""
    cells = [
        mk_cell("a", (0, 0, 100, 30), role="cell"),
        mk_cell("b", (2, 1, 50, 20), role="cell"),
    ]
    out, remap = sort_cells(cells)

    assert sorted(remap.values()) == ["r0c0", "r0c1"]
    assert len({c.id for c in out}) == 2


def test_sort_cells_column_ordinal_is_scoped_to_the_row():
    """列は行内序数: 行ごとに区切りが違ってもセル増減の影響が行内に閉じる"""
    base = [
        mk_cell("a", (0, 0, 50, 30), role="cell"),
        mk_cell("b", (200, 0, 300, 30), role="cell"),
        mk_cell("c", (0, 60, 120, 90), role="cell"),
        mk_cell("d", (120, 60, 300, 90), role="cell"),
    ]
    _, remap1 = sort_cells([mk_cell(c.id, c.box, role=c.role) for c in base])

    # 行0 の先頭にセルが1つ増えても、行1 の ID は変わらない
    extra = [mk_cell("x", (60, 0, 150, 30), role="cell")] + base
    _, remap2 = sort_cells([mk_cell(c.id, c.box, role=c.role) for c in extra])

    assert remap1["c"] == remap2["c"]
    assert remap1["d"] == remap2["d"]
    # 行0 は挿入位置より右だけがずれる
    assert remap1["a"] == remap2["a"] == "r0c0"
    assert remap1["b"] == "r0c1" and remap2["b"] == "r0c2"


def test_sort_elements_assigns_prefix_ids_in_sorted_order():
    e0 = mk_elem(None, (0, 0, 10, 10))
    e1 = mk_elem(None, (0, 20, 10, 30))
    e2 = mk_elem(None, (20, 0, 30, 10))

    out = _sort_elements([e1, e2, e0], prefix="t")
    assert [e.id for e in out] == ["t0", "t1", "t2"]

    # min_height=10 -> key=(y//10, x)
    # e0: (0,0), e2:(0,20), e1:(2,0)
    assert [e.box for e in out] == [(0, 0, 10, 10), (20, 0, 30, 10), (0, 20, 10, 30)]

    out = _sort_elements([], prefix="x")
    assert out == []


def test_assign_ids_remaps_grid_and_kv_consistently():
    # cells dict: old ids
    cA = mk_cell("A", (0, 0, 10, 10), role="cell")
    cB = mk_cell("B", (10, 0, 20, 10), role="cell")
    cC = mk_cell("C", (0, 10, 10, 20), role="cell")

    # grid / kv の最低限
    grid = SimpleNamespace(
        id=None,
        data=[["A", "B"], ["C", None]],
        col_headers=[["A"], ["B"]],
        box=(0, 0, 20, 20),
    )
    kv = SimpleNamespace(id=None, key=["A"], value="B")

    table_information = {
        "grids": [grid],
        "kv_items": [kv],
        "cells": {"A": cA, "B": cB, "C": cC},
    }

    _assign_ids(table_information)

    # grid/kv id がつく
    assert table_information["grids"][0].id == "g0"
    assert table_information["kv_items"][0].id == "kv0"

    # cells が remap 後 dict になっている（r{行}c{列}）
    new_ids = set(table_information["cells"].keys())
    assert all(cid.startswith("r") for cid in new_ids)

    # grid.data / col_headers / kv.key/value も remap されている
    assert all(
        x is None or x.startswith("r")
        for row in table_information["grids"][0].data
        for x in row
    )
    assert all(
        x is None or x.startswith("r")
        for col in table_information["grids"][0].col_headers
        for x in col
    )
    assert all(k.startswith("r") for k in table_information["kv_items"][0].key)
    assert table_information["kv_items"][0].value.startswith("r")


# ============================================================
# 領域とセルの対応 / 領域重複の解決
# ============================================================
def _mk_region(rid, box, role="kv_item", score=1.0):
    from yomitoku.schemas.table_semantic_parser import RegionSchema

    return RegionSchema(id=rid, box=list(box), role=role, score=score)


def test_region_cell_ids_returns_contained_cells():
    from yomitoku.table_semantic_parser import _region_cell_ids

    cells = [
        mk_cell("inside", (10, 10, 90, 40)),
        mk_cell("outside", (200, 200, 300, 240)),
        mk_cell("partial", (80, 10, 200, 40)),  # 領域内は一部のみ
    ]
    region = _mk_region("r0", (0, 0, 100, 50))

    ids = _region_cell_ids(region, cells)

    assert "inside" in ids
    assert "outside" not in ids
    # threshold=0.5: 内包率が半分未満のセルは含まれない
    assert "partial" not in ids


def test_resolve_grid_kv_conflict_prefers_higher_cell_coverage():
    """gridとkvが同じセル群を奪い合う場合、セル網羅度が高い方が残る"""
    from yomitoku.table_semantic_parser import _resolve_overlapping_regions

    cells = [mk_cell(f"c{i}", (i * 100, 0, (i + 1) * 100, 50)) for i in range(4)]

    # grid はセル4つ、kv はそのうち1つだけを覆う
    grid = _mk_region("g", (0, 0, 400, 50), role="grid")
    kv = _mk_region("k", (0, 0, 100, 50), role="kv_item")

    grids, kvs = _resolve_overlapping_regions([grid], [kv], cells)

    assert [g.id for g in grids] == ["g"]
    assert kvs == []


def test_resolve_grid_kv_conflict_drops_grid_when_kv_covers_more():
    from yomitoku.table_semantic_parser import _resolve_overlapping_regions

    cells = [mk_cell(f"c{i}", (i * 100, 0, (i + 1) * 100, 50)) for i in range(4)]

    grid = _mk_region("g", (0, 0, 100, 50), role="grid")
    kv = _mk_region("k", (0, 0, 400, 50), role="kv_item")

    grids, kvs = _resolve_overlapping_regions([grid], [kv], cells)

    assert grids == []
    assert [k.id for k in kvs] == ["k"]


def test_resolve_non_conflicting_grid_and_kv_both_kept():
    from yomitoku.table_semantic_parser import _resolve_overlapping_regions

    cells = [
        mk_cell("a", (0, 0, 100, 50)),
        mk_cell("b", (0, 100, 100, 150)),
    ]
    grid = _mk_region("g", (0, 0, 100, 50), role="grid")
    kv = _mk_region("k", (0, 100, 100, 150), role="kv_item")

    grids, kvs = _resolve_overlapping_regions([grid], [kv], cells)

    assert len(grids) == 1 and len(kvs) == 1


# ============================================================
# is_grid_cluster
# ============================================================
def test_is_grid_cluster_true_for_2x2():
    from yomitoku.table_semantic_parser import is_grid_cluster

    nodes = {
        "header": [],
        "cell": [
            mk_cell("a", (0, 0, 100, 50)),
            mk_cell("b", (100, 0, 200, 50)),
            mk_cell("c", (0, 50, 100, 100)),
            mk_cell("d", (100, 50, 200, 100)),
        ],
        "empty": [],
    }

    assert is_grid_cluster(nodes) is True


def test_is_grid_cluster_false_for_single_row():
    from yomitoku.table_semantic_parser import is_grid_cluster

    nodes = {
        "header": [],
        "cell": [
            mk_cell("a", (0, 0, 100, 50)),
            mk_cell("b", (100, 0, 200, 50)),
            mk_cell("c", (200, 0, 300, 50)),
        ],
        "empty": [],
    }

    assert is_grid_cluster(nodes) is False


# ============================================================
# TableSemanticParser.aggregate (OCR単語のセルへの集約)
# ============================================================
def _mk_word(content, box, direction="horizontal"):
    from yomitoku.schemas import WordPrediction

    x1, y1, x2, y2 = box
    return WordPrediction(
        points=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        content=content,
        direction=direction,
        rec_score=0.99,
        det_score=0.99,
    )


def test_aggregate_assigns_words_to_best_cell_in_reading_order():
    from yomitoku.schemas import OCRSchema
    from yomitoku.table_semantic_parser import TableSemanticParser

    cell_a = mk_cell("a", (0, 0, 200, 50))
    cell_b = mk_cell("b", (0, 100, 200, 150))
    ocr = OCRSchema(
        words=[
            _mk_word("世界", (100, 10, 190, 40)),  # 意図的に逆順
            _mk_word("こんにちは", (10, 10, 100, 40)),
        ]
    )

    TableSemanticParser.aggregate(object(), ocr, [cell_a, cell_b])

    # 読み順 (左→右) に結合される
    assert cell_a.contents == "こんにちは世界"
    # 単語が無いセルは空文字
    assert cell_b.contents == ""


def test_aggregate_skips_group_cells_and_low_overlap_words():
    from yomitoku.schemas import OCRSchema
    from yomitoku.table_semantic_parser import TableSemanticParser

    group = mk_cell("g", (0, 0, 200, 50), role="group", contents="before")
    cell = mk_cell("a", (0, 100, 200, 150))
    ocr = OCRSchema(
        words=[
            _mk_word("グループ内", (10, 10, 100, 40)),  # group にのみ重なる
            _mk_word("枠外", (500, 500, 600, 530)),  # どのセルにも重ならない
        ]
    )

    TableSemanticParser.aggregate(object(), ocr, [group, cell])

    # group セルには割り当てられない (contents は空にリセットされる)
    assert group.contents == ""
    assert cell.contents == ""


# ============================================================
# TableSemanticParser.replace_table_to_paragraphs
# ============================================================
def test_replace_table_to_paragraphs_converts_sparse_tables():
    from yomitoku.table_semantic_parser import TableSemanticParser

    sparse = SimpleNamespace(
        box=[0, 0, 100, 50],
        cells=[mk_cell("a", (0, 0, 100, 50), role="cell")],
    )
    dense = SimpleNamespace(
        box=[0, 100, 200, 200],
        cells=[
            mk_cell("h", (0, 100, 100, 150), role="header"),
            mk_cell("v", (100, 100, 200, 150), role="cell"),
        ],
    )
    paragraphs = []

    kept = TableSemanticParser.replace_table_to_paragraphs(
        object(), [sparse, dense], paragraphs
    )

    # セル数2未満のテーブルは段落に変換され、テーブルからは除外される
    assert kept == [dense]
    assert len(paragraphs) == 1
    assert paragraphs[0].box == [0, 0, 100, 50]


# ============================================================
# 可視化 (kv_items_visualizer / dag_visualizer)
# ============================================================
def test_kv_items_visualizer_draws_green_arrows():
    import numpy as np

    from yomitoku.table_semantic_parser import kv_items_visualizer

    table = SimpleNamespace(
        cells={
            "k": mk_cell("k", (10, 10, 50, 40)),
            "v": mk_cell("v", (50, 10, 120, 40)),
        },
        kv_items=[SimpleNamespace(key=["k"], value="v")],
    )
    img = np.full((100, 200, 3), 255, dtype=np.uint8)

    out = kv_items_visualizer(table, img)

    # 緑 (BGR=(0,255,0)) の矢印が描画されている
    green = (out[:, :, 0] == 0) & (out[:, :, 1] == 255) & (out[:, :, 2] == 0)
    assert green.any()


def test_kv_items_visualizer_ignores_missing_cells():
    import numpy as np

    from yomitoku.table_semantic_parser import kv_items_visualizer

    table = SimpleNamespace(
        cells={},
        kv_items=[SimpleNamespace(key=["missing"], value="also_missing")],
    )
    img = np.full((50, 50, 3), 255, dtype=np.uint8)

    out = kv_items_visualizer(table, img)

    assert (out == 255).all()


def test_dag_visualizer_draws_edges():
    import networkx as nx
    import numpy as np

    from yomitoku.table_semantic_parser import dag_visualizer

    dag = nx.DiGraph()
    dag.add_node("a", bbox=[10, 10, 50, 40])
    dag.add_node("b", bbox=[50, 10, 120, 40])
    dag.add_edge("a", "b", dir="R")
    dag.add_edge("b", "a", dir="L")  # L/U は描画されない
    img = np.full((100, 200, 3), 255, dtype=np.uint8)

    out = dag_visualizer(dag, img)

    assert (out != 255).any()


# ============================================================
# TableSemanticParser.__call__ (モデル非依存: run_models を差し替え)
# ============================================================
def _mk_parser():
    from yomitoku.table_semantic_parser import TableSemanticParser

    parser = TableSemanticParser.__new__(TableSemanticParser)
    parser.visualize = False
    parser.merge_same_column_values = False
    return parser


def _mk_detector_table(box, cells, kv_regions=(), grid_regions=()):
    from yomitoku.schemas.table_semantic_parser import TableDetectorSchema

    return TableDetectorSchema(
        id=None,
        box=list(box),
        role=None,
        cells=list(cells),
        kv_regions=list(kv_regions),
        grid_regions=list(grid_regions),
    )


def _mk_real_cell(cid, box, role="cell", contents=""):
    from yomitoku.schemas.table_semantic_parser import CellSchema

    return CellSchema(
        meta={},
        id=cid,
        box=list(box),
        role=role,
        contents=contents,
        row=None,
        col=None,
        row_span=None,
        col_span=None,
    )


def _call_with_fake_models(parser, monkeypatch, tables, paragraphs, words=(), **kwargs):
    import numpy as np

    from yomitoku.schemas import OCRSchema

    async def _fake_run_models(_img):
        return OCRSchema(words=list(words)), tables, paragraphs

    monkeypatch.setattr(parser, "run_models", _fake_run_models)
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    return parser(img, **kwargs)


def test_call_builds_schema_with_sorted_ids(monkeypatch):
    from yomitoku.schemas import Element

    parser = _mk_parser()

    key = _mk_real_cell("k", (0, 0, 60, 30), role="header", contents="項目")
    value = _mk_real_cell("v", (60, 0, 180, 30), role="cell", contents="値")
    table = _mk_detector_table(
        (0, 0, 180, 30),
        [key, value],
        kv_regions=[_mk_region("r0", (0, 0, 180, 30))],
    )
    paragraph = Element(
        id=None, box=[0, 100, 100, 130], score=0.9, role=None, contents="本文"
    )
    # セルのcontentsはOCR単語のaggregateで決まる
    words = [
        _mk_word("項目", (5, 5, 55, 25)),
        _mk_word("値", (65, 5, 175, 25)),
    ]

    semantic_info, vis_layout, vis_ocr = _call_with_fake_models(
        parser, monkeypatch, [table], [paragraph], words=words
    )

    assert [t.id for t in semantic_info.tables] == ["t0"]
    assert [p.id for p in semantic_info.paragraphs] == ["p0"]

    t0 = semantic_info.tables[0]
    assert [kv.id for kv in t0.kv_items] == ["kv0"]
    # キー/値のテキストが解決されている
    kv = t0.kv_items[0]
    keys = [kv.key] if isinstance(kv.key, str) else kv.key
    assert [t0.safe_contents(k) for k in keys] == ["項目"]
    assert t0.safe_contents(kv.value) == "値"

    # visualize=False でも入力画像と同形状のコピーが返る
    assert vis_layout.shape == (64, 64, 3)
    assert vis_ocr.shape == (64, 64, 3)


def test_call_kv_only_ignores_grid_regions(monkeypatch):
    parser = _mk_parser()

    key = _mk_real_cell("k", (0, 0, 60, 30), role="header", contents="項目")
    value = _mk_real_cell("v", (60, 0, 180, 30), role="cell", contents="値")
    table = _mk_detector_table(
        (0, 0, 180, 30),
        [key, value],
        kv_regions=[_mk_region("r0", (0, 0, 180, 30))],
        grid_regions=[_mk_region("g0", (0, 0, 180, 30), role="grid")],
    )

    semantic_info, _, _ = _call_with_fake_models(
        parser, monkeypatch, [table], [], kv_only=True
    )

    assert semantic_info.tables[0].grids == []
    assert len(semantic_info.tables[0].kv_items) == 1
