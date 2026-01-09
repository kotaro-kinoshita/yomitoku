from .schemas.table_semantic_parser import (
    KvItemSchema,
    TableGridCell,
    TableGridRow,
    TableGridSchema,
)


def walk_to_terminal_by_edge_type_bbox_axis(
    G,
    start,
    edge_attr: str,
    edge_value,
    bbox_attr: str = "bbox",
    axis: str = "x",  # "x" or "y"
    use: str = "center",  # "center" or "min" or "max"
    max_steps: int = 10_000,
):
    """
    特定タイプの out edge だけ辿る。
    分岐したら bbox の x もしくは y の差が最小の次ノードを選ぶ（greedy）。
    そのタイプの出エッジが無くなったら終了して path を返す。
    """
    assert axis in ("x", "y")
    assert use in ("center", "min", "max")

    def axis_value(node):
        bbox = G.nodes[node].get(bbox_attr)
        if bbox is None:
            raise ValueError(f"Node {node} lacks '{bbox_attr}'")
        x0, y0, x1, y1 = bbox
        if axis == "x":
            return (x0 + x1) / 2 if use == "center" else (x0 if use == "min" else x1)
        else:
            return (y0 + y1) / 2 if use == "center" else (y0 if use == "min" else y1)

    initial = start
    path = [start]
    cur = start
    seen = {start}
    for _ in range(max_steps):
        # 先頭ノードの位置を基準にする
        # -> 例えば上方向に辿る場合、各ノードの y 座標との差が最小のノードを選ぶため
        cur_val = axis_value(initial)

        candidates = []
        for _, v, data in G.out_edges(cur, data=True):
            if data.get(edge_attr) != edge_value:
                continue
            v_val = axis_value(v)
            dist = abs(v_val - cur_val)

            candidates.append((dist, v))

        if not candidates:
            break

        candidates.sort(key=lambda t: t[0])
        nxt = candidates[0][1]

        # サイクル対策（必要ならここは例外にする等に変更）
        if nxt in seen:
            path.append(nxt)
            break

        path.append(nxt)
        seen.add(nxt)
        cur = nxt

    return path


def _get_kv_items_groups(match, nodes, group_direction):
    rows = [
        group
        for group in nodes["group"]
        if group.id in group_direction and group_direction[group.id] == "H"
    ]

    group_to_cells = match["group_to_cells"]
    cell_to_grid_region = match.get("cell_to_grid_region", {})

    # Gridに属しているセルが含まれるグループは除外
    kept_rows = []
    for row in rows:
        cells = group_to_cells.get(row.id, [])
        for cell in cells:
            if cell in cell_to_grid_region:
                break
        else:
            kept_rows.append(row)
    return kept_rows


def _get_grid_n_rows(dag, match, grid_region):
    """グリッド領域内で最も行数の多い列を基準列として取得"""

    n_rows = 0
    base_cols = []
    group_to_cells = match["group_to_cells"]

    for group_id in grid_region["member_ids"]:
        cells = group_to_cells.get(group_id, [])

        for cell_id in cells:
            nodes = walk_to_terminal_by_edge_type_bbox_axis(
                dag,
                cell_id,
                edge_attr="dir",
                edge_value="D",
                axis="y",
                use="center",
            )

            cell_nodes = [
                node for node in nodes if dag.nodes[node]["role"] in ["cell", "empty"]
            ]

            if n_rows < len(cell_nodes):
                base_cols = cell_nodes
                n_rows = len(cell_nodes)

    return n_rows, base_cols


def _calc_kv_items(dag, match, groups, scanned_nodes):
    """
    グループ内の終端セルから左方向に走査してキーとバリューを取得
    """

    group_to_cells = match["group_to_cells"]
    kv_items = []

    is_worked_cell = []
    for group in groups:
        cells = group_to_cells.get(group.id, None)

        if cells is None:
            continue

        for cell in cells:
            is_not_reef = any(
                data.get("dir") == "R" for _, _, data in dag.out_edges(cell, data=True)
            )

            if is_not_reef:
                continue

            if cell in is_worked_cell:
                continue

            is_worked_cell.append(cell)

            search_left = walk_to_terminal_by_edge_type_bbox_axis(
                dag,
                cell,
                edge_attr="dir",
                edge_value="L",
                axis="y",
                use="center",
            )

            key = []
            value = []
            for node in search_left:
                if dag.nodes[node]["role"] in ["cell", "empty"]:
                    value.append(dag.nodes[node])

                elif dag.nodes[node]["role"] == "header":
                    key.append(dag.nodes[node])

            key = sorted(key, key=lambda x: x["bbox"][0])
            value = sorted(value, key=lambda x: x["bbox"][0])

            for v in value:
                scanned_nodes[v["id"]] = True

            kv_items.append(
                KvItemSchema(
                    key=[k["id"] for k in key],
                    value=[v["id"] for v in value],
                )
            )

    for node_id, scanned in scanned_nodes.items():
        if not scanned:
            search_left = walk_to_terminal_by_edge_type_bbox_axis(
                dag,
                node_id,
                edge_attr="dir",
                edge_value="L",
                axis="y",
                use="center",
            )
            key = []
            value = []
            for node in search_left:
                if dag.nodes[node]["role"] in ["cell", "empty"]:
                    value.append(dag.nodes[node])
                elif dag.nodes[node]["role"] == "header":
                    key.append(dag.nodes[node])
            key = sorted(key, key=lambda x: x["bbox"][0])
            value = sorted(value, key=lambda x: x["bbox"][0])
            for v in value:
                scanned_nodes[v["id"]] = True
            kv_items.append(
                KvItemSchema(
                    key=[k["id"] for k in key],
                    value=[v["id"] for v in value],
                )
            )

    return kv_items


def _calc_grids(dag, match, grid_regions, scanned_nodes):
    """
    基準列のセルから左右に走査して行全体を取得し、各セルから上方向に辿って列ヘッダーを取得
    """

    grids = []
    for j, grid_region in enumerate(grid_regions):
        # 最も行数の多い列を基準列として取得
        n_rows, base_cols = _get_grid_n_rows(dag, match, grid_region)
        base_cols = sorted(base_cols, key=lambda x: dag.nodes[x]["bbox"][1])
        rows = []

        for cell in base_cols:
            # 基準列のセルから左右に辿って行全体を取得
            search_right = walk_to_terminal_by_edge_type_bbox_axis(
                dag,
                cell,
                edge_attr="dir",
                edge_value="R",
                axis="y",
                use="center",
            )

            search_left = walk_to_terminal_by_edge_type_bbox_axis(
                dag,
                cell,
                edge_attr="dir",
                edge_value="L",
                axis="y",
                use="center",
            )
            row_cells = set(search_right) | set(search_left)
            row_cells = sorted(row_cells, key=lambda x: dag.nodes[x]["bbox"][0])

            row_header = [
                node for node in row_cells if dag.nodes[node]["role"] == "header"
            ]

            row_value = [
                node
                for node in row_cells
                if dag.nodes[node]["role"] in ["cell", "empty"]
            ]

            col_headers = []

            # 各セルから上方向に辿って列ヘッダーを取得
            for value in row_value:
                search_top = walk_to_terminal_by_edge_type_bbox_axis(
                    dag,
                    value,
                    edge_attr="dir",
                    edge_value="U",
                    axis="x",
                    use="min",
                )

                col_header = [
                    node for node in search_top if dag.nodes[node]["role"] == "header"
                ]

                col_headers.append(
                    sorted(
                        col_header,
                        key=lambda x: dag.nodes[x]["bbox"][1],
                    )
                )

            row = []
            for col_header, v in zip(col_headers, row_value):
                col_header = [dag.nodes[h]["id"] for h in col_header]
                row_header = [dag.nodes[h]["id"] for h in row_header]
                row.append(
                    TableGridCell(
                        row_keys=row_header,
                        col_keys=col_header,
                        value=dag.nodes[v]["id"],
                    )
                )

                scanned_nodes[v] = True

            rows.append(TableGridRow(cells=row, id=f"r{str(len(rows))}"))
        grids.append(TableGridSchema(rows=rows, id=None))
    return grids, scanned_nodes


def _convert_grid_to_kv_items(grids, kv_items):
    """一行のみもしくは一列のみのグリッドをKVアイテムに変換"""

    kept_grids = []
    for grid in grids:
        if len(grid.rows) == 1:
            row = grid.rows[0]
            for cell in row.cells:
                kv_items.append(
                    KvItemSchema(
                        key=cell.row_keys + cell.col_keys,
                        value=[cell.value],
                    )
                )
            continue

        is_single_col = True
        for row in grid.rows:
            if len(row.cells) > 1:
                is_single_col = False
                break

        if is_single_col:
            for row in grid.rows:
                for cell in row.cells:
                    kv_items.append(
                        KvItemSchema(
                            key=cell.row_keys + cell.col_keys,
                            value=[cell.value],
                        )
                    )
            continue

        grid.id = f"g{str(len(kept_grids))}"
        kept_grids.append(grid)

    return kept_grids, kv_items


def parse_semantic_table_information(
    cell_relation_dag, match, nodes, grid_regions, group_direction
):
    scanned_nodes = {n.id: False for n in nodes["cell"] + nodes["empty"]}

    grids, scanned_nodes = _calc_grids(
        cell_relation_dag, match, grid_regions, scanned_nodes
    )

    kv_items_groups = _get_kv_items_groups(match, nodes, group_direction)

    kv_items = _calc_kv_items(cell_relation_dag, match, kv_items_groups, scanned_nodes)

    grids, kv_items = _convert_grid_to_kv_items(grids, kv_items)

    return grids, kv_items
