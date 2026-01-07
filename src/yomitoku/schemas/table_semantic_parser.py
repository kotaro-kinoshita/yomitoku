import re
import json

from ..base import BaseSchema

from typing import List, Union
from pydantic import conlist, Field, PrivateAttr

from typing import Dict
from pandas import DataFrame
import pandas as pd

from .document_analyzer import WordPrediction


from collections import defaultdict


def make_unique_all(seq):
    counter = defaultdict(int)
    result = []

    for x in seq:
        key = tuple(x)
        idx = counter[key]
        result.append(x + [idx])
        counter[key] += 1

    for res, x in zip(result, seq):
        if counter[tuple(x)] == 1:
            res.pop()

    return result


def normalize(text: str) -> str:
    """空白（全角・半角）を除去"""
    return re.sub(r"[ 　]", "", text)


class CellSchema(BaseSchema):
    contents: Union[str, None] = Field(
        ...,
        description="Text content of the cell",
    )
    role: Union[str, None] = Field(
        ...,
        description="Role of the cell, e.g., ['cell', 'header', 'empty', 'group']",
    )
    id: Union[str, None] = Field(
        ...,
        description="Unique identifier of the cell",
    )
    box: conlist(int, min_length=4, max_length=4) = Field(
        ...,
        description="Bounding box of the cell in the format [x1, y1, x2, y2]",
    )


class CellDetectorSchema(BaseSchema):
    id: Union[str, None] = Field(
        ...,
        description="Unique identifier of the cell",
    )
    box: conlist(int, min_length=4, max_length=4) = Field(
        ...,
        description="Bounding box of the cell in the format [x1, y1, x2, y2]",
    )
    role: Union[str, None] = Field(
        ...,
        description="Role of the cell, e.g., ['cell', 'header', 'empty', 'group']",
    )
    cells: List[CellSchema] = Field(
        ...,
        description="List of detected table cells",
    )


class KvItemSchema(BaseSchema):
    key: List[str] = (
        Field(
            ...,
            description="Key of the key-value kv_items",
        ),
    )
    value: List[str] = Field(
        ...,
        description="Value of the key-value kv_items",
    )


class TableGridCell(BaseSchema):
    row_keys: List[str]
    col_keys: List[str]
    value: str


class TableGridRow(BaseSchema):
    id: Union[str, None] = Field(
        ...,
        description="Unique identifier of the table grid row",
    )
    cells: List[TableGridCell] = Field(
        ...,
        description="List of cells in the table grid row",
    )


class TableGridSchema(BaseSchema):
    id: Union[str, None] = Field(
        ...,
        description="Unique identifier of the table grid",
    )
    rows: List[TableGridRow] = Field(
        ...,
        description="List of rows in the table grid",
    )


class TableSemanticContentsSchema(BaseSchema):
    id: str | None = Field(None, description="Unique identifier of the table")
    style: str = Field(
        ..., description="Border style of the table, e.g., ['border', 'borderless']"
    )
    box: conlist(int, min_length=4, max_length=4) = Field(
        ..., description="Bounding box [x1, y1, x2, y2]"
    )

    cells: Dict[str, CellSchema] = Field(..., description="Cells keyed by cell_id")
    kv_items: List[KvItemSchema] = Field(
        ..., description="Key-value items extracted from the table"
    )
    grids: List[TableGridSchema] = Field(
        ..., description="Grid representation of the table"
    )

    _export: "TableSemanticContentsExport" = PrivateAttr()
    _view: "TableSemanticContentsView" = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._view = TableSemanticContentsView(self)
        self._export = TableSemanticContentsExport(self)

    @property
    def view(self) -> "TableSemanticContentsView":
        return self._view

    @property
    def export(self) -> "TableSemanticContentsExport":
        return self._export

    def safe_contents(self, cell_id: str) -> str:
        c = self.cells.get(cell_id)
        return c.contents or "" if c is not None else ""

    def search_cell_by_id(self, cell_id: str) -> CellSchema | None:
        """
        Search for a cell by its ID.
        セルIDに対応するセルを返す
        Args:
            cell_id (str): 検索するセルID

        """

        return self.cells.get(str(cell_id))

    def search_cell_by_query(self, query: str) -> List[CellSchema]:
        """Search for cells containing the query string in their contents.
        クエリーに部分一致する内容を持つセルを返す

        Args:
            query (str): 検索するクエリ文字列. セルの内容に部分一致するものを検索
        """

        q = normalize(query)
        out: List[CellSchema] = []
        for cell in self.cells.values():
            if not cell.contents:
                continue
            if cell.role == "group":
                continue
            if q in normalize(cell.contents):
                out.append(cell)
        return out

    def search_kv_items_by_key(self, query: str) -> List[dict]:
        """
        Search for key-value items or grid cells where the key matches the query string.
        クエリーに部分一致するキーを持つKVアイテムおよびグリッドセルを返す
        Args:
            query (str): 検索するクエリ文字列. キー部分に部分一致するものを検索
        """

        q = normalize(query)
        results: List[dict] = []

        # kv_items 側
        for kv_item in self.kv_items:
            key_cells = [self.cells.get(k) for k in kv_item.key]
            val_cells = [self.cells.get(v) for v in kv_item.value]

            key_text = "".join([(kc.contents or "") for kc in key_cells if kc])
            if q in normalize(key_text):
                results.append({"key": key_cells, "value": val_cells})

        # grids 側
        for grid in self.grids:
            for row in grid.rows:
                for cell_info in row.cells:
                    row_cells = [self.cells.get(rk) for rk in cell_info.row_keys]
                    col_cells = [self.cells.get(ck) for ck in cell_info.col_keys]
                    v_cell = self.cells.get(cell_info.value)

                    row_text = "".join([(c.contents or "") for c in row_cells if c])
                    col_text = "".join([(c.contents or "") for c in col_cells if c])

                    if q in normalize(row_text):
                        results.append({"key": row_cells, "value": [v_cell]})
                    elif q in normalize(col_text):
                        results.append({"key": col_cells, "value": [v_cell]})

        return results

    def search_table_by_column_name(
        self,
        queries,
    ) -> "TableSemanticContentsSchema":
        """
        Search for columns matching the specified column names.
        指定した列名に部分一致する列のみを含むテーブルを返す

        Args:
            queries (List[str]): 検索する列名のリスト
        """

        table_contents = {
            "id": self.id,
            "box": self.box,
            "style": self.style,
            "cells": self.cells,
            "grids": [],
            "kv_items": self.kv_items,
        }
        for grid in self.grids:
            filter_grid = self.filter_columns_ignore_space(
                grid,
                queries,
            )
            if filter_grid is not None:
                table_contents["grids"].append(filter_grid)
        return TableSemanticContentsSchema(**table_contents)

    def filter_columns_ignore_space(
        self,
        grid,
        queries,
    ) -> Union[TableGridSchema, None]:
        """
        列名から空白を除去した上で、
        queries のいずれかに部分一致する列のみ残す
        """
        norm_queries = [normalize(q) for q in queries]
        result = {
            "id": grid.id,
            "rows": [],
        }
        for row in grid.rows:
            filtered_row = []
            for cell in row.cells:
                has_query = False
                key_contents = [
                    self.cells.get(rk).contents or "" for rk in cell.col_keys
                ]
                nk = normalize("".join(key_contents))
                for q in norm_queries:
                    if q in nk:
                        has_query = True
                        break
                if has_query:
                    filtered_row.append(cell)
            if filtered_row:
                result["rows"].append(
                    {
                        "id": row.id,
                        "cells": filtered_row,
                    }
                )
        return TableGridSchema(**result) if result["rows"] else None  #


class TableSemanticContentsExport:
    def __init__(self, table: TableSemanticContentsSchema):
        self.table = table

    def grids_to_dataframe(self, grid_id: str = "0") -> DataFrame | None:
        grids = self.table.view.grids_to_dicts()
        for g in grids:
            if str(g["id"]) != str(grid_id):
                continue
            return pd.DataFrame(g["rows"])
        return None

    def grids_to_csv(self, out_path, columns=None, grid_id="0"):
        """
        Export table grids to a CSV file.
        テーブルグリッドの内容をCSV形式で出力
        Args:
            out_path (str): 出力先のファイルパス
            columns (List[str], optional): 抽出する列名のリスト. 指定しない場合は全列を抽出
            grid_id (str, optional): グリッドID. デフォルトは "0"
        """
        table_contents = self.table
        if columns is not None:
            table_contents = self.table.search_table_by_column_name(queries=columns)

        df = table_contents.export.grids_to_dataframe(grid_id=grid_id)
        if df is not None:
            return df.to_csv(out_path)
        return None

    def kv_items_to_dataframe(self) -> DataFrame:
        """
        Convert key-value items to a pandas DataFrame.
        キーと値のペアの文字列を展開し、DataFrame形式に変換
        """

        return pd.DataFrame([self.table.view.kv_items_to_dict()])

    def kv_items_to_csv(self, out_path: str) -> str:
        """
        Export key-value items to a CSV file.
        キーと値のペアの文字列を展開し、CSV形式で出力
        Args:
            out_path (str): 出力先のファイルパス
        """

        return self.kv_items_to_dataframe().to_csv(out_path, index=False)

    def grids_to_json(self, out_path) -> str:
        """
        Export table grids to a JSON file.
        テーブルグリッドの内容をJSON形式で出力
        Args:
            out_path (str): 出力先のファイルパス
        """

        grids = self.table.view.grids_to_dicts()
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(grids, f, ensure_ascii=False, indent=4)

    def kv_items_to_json(self, out_path) -> str:
        """
        Export key-value items to a JSON file.
        キーと値のペアの文字列を展開し、JSON形式で出力
        Args:
            out_path (str): 出力先のファイルパス
        """

        kv_items = self.table.view.kv_items_to_dict()
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(kv_items, f, ensure_ascii=False, indent=4)


class TableSemanticContentsView:
    def __init__(self, table: TableSemanticContentsSchema):
        self.table = table

    def kv_items_to_dict(self) -> dict:
        """
        Convert key-value items to a dictionary.
        キーと値のペアの文字列を展開し、辞書形式に変換
        """

        t = self.table
        parsed = {}
        keys, vals = [], []

        for kv in t.kv_items:
            k = [t.safe_contents(i) for i in kv.key]
            v = [t.safe_contents(i) for i in kv.value]
            keys.append(k)
            vals.append(v)

        keys = make_unique_all(keys)
        for k, v in zip(keys, vals):
            parsed["_".join(map(str, k))] = " ".join(map(str, v))

        return parsed

    def grids_to_dicts(self) -> list[dict]:
        """
        Convert table grids to a list of dictionaries.
        テーブルグリッドの内容を辞書形式に変換
        """

        t = self.table
        results = []
        for grid in t.grids:
            row_record_list = []
            for row in grid.rows:
                parsed_row = {}
                col_key_list, value_list = [], []
                row_key = None

                for cell in row.cells:
                    rk = [t.safe_contents(i) for i in cell.row_keys]
                    ck = [t.safe_contents(i) for i in cell.col_keys]
                    v = t.safe_contents(cell.value)

                    if row_key is None and rk:
                        row_key = rk
                    col_key_list.append(ck)
                    value_list.append(v)

                row_key = row_key or []
                col_key_list = make_unique_all(col_key_list)

                for i, rk in enumerate(row_key):
                    parsed_row[f"row_key_{i}"] = rk

                for ck, v in zip(col_key_list, value_list):
                    parsed_row["_".join(map(str, ck))] = v

                row_record_list.append(parsed_row)
            results.append({"id": grid.id, "rows": row_record_list})

        return results


class TableSemanticParserSchema(BaseSchema):
    tables: List[TableSemanticContentsSchema] = Field(
        ...,
        description="List of tables with semantic information",
    )

    words: List[WordPrediction] = Field(
        ...,
        description="List of recognized words in the document",
    )

    def search_table_by_id(
        self, table_id: str
    ) -> Union[TableSemanticContentsSchema, None]:
        """Search for a table by its ID.
        Args:
            table_id (str): 検索するテーブルID
        """
        for table in self.tables:
            if table.id == str(table_id):
                return table
