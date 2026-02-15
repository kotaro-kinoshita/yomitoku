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
    assert nodes["group"] == []


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


def test_sort_cells_remaps_ids_and_orders_values_before_groups():
    # min_height=10 として並びが安定するように作る
    c0 = mk_cell("old0", (0, 0, 10, 10), role="cell")
    c1 = mk_cell("old1", (20, 0, 30, 10), role="header")
    c2 = mk_cell("old2", (0, 20, 10, 30), role="empty")
    g0 = mk_cell("grp", (0, 5, 50, 60), role="group")

    cells, remap = sort_cells([g0, c2, c0, c1])

    # values(cell/header/empty) が先、groups が後
    assert [c.role for c in cells[:-1]] == ["cell", "header", "empty"]
    assert cells[-1].role == "group"

    # remap できてる
    assert set(remap.keys()) == {"old0", "old1", "old2", "grp"}
    assert all(c.id.startswith("c") for c in cells)

    cells, remap = sort_cells([])
    assert cells == []
    assert remap == {}


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

    # cells が remap 後 dict になっている（c0..）
    new_ids = set(table_information["cells"].keys())
    assert all(cid.startswith("c") for cid in new_ids)

    # grid.data / col_headers / kv.key/value も remap されている
    # （正確な c番号は sort_cells の並び依存なので “prefix” で判定）
    assert all(
        x is None or x.startswith("c")
        for row in table_information["grids"][0].data
        for x in row
    )
    assert all(
        x is None or x.startswith("c")
        for col in table_information["grids"][0].col_headers
        for x in col
    )
    assert all(k.startswith("c") for k in table_information["kv_items"][0].key)
    assert table_information["kv_items"][0].value.startswith("c")
