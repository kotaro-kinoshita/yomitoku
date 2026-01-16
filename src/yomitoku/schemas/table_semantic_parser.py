import re
import json

from ..base import BaseSchema

import numpy as np

from typing import List, Union, Optional, Literal, Any
from pydantic import conlist, Field, PrivateAttr, field_validator, field_serializer

from typing import Dict
from pandas import DataFrame
import pandas as pd

from .document_analyzer import WordPrediction, ParagraphSchema
from ..reading_order import prediction_reading_order
from ..utils.visualizer import kv_items_visualizer, grids_visualizer

from collections import defaultdict


from ..utils.misc import (
    is_contained,
    is_bottom_adjacent,
    is_right_adjacent,
    calc_overlap_ratio,
    quad_to_xyxy,
)

MatchPolicy = Literal["cell_id", "bbox"]


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


class TemplateMetaSchema(BaseSchema):
    template_version: str = Field("beta", description="Template schema version")
    template_id: str | None = Field(None, description="Human-readable template id")
    notes: str | None = Field(None, description="Notes for template editors")

    match_policy: MatchPolicy = Field("cell_id", description="How to match cells")


class CellSchema(BaseSchema):
    meta: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for template/semantics"
    )

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
    row: Union[int, None] = Field(
        ...,
        description="Row index of the cell in the table",
    )

    col: Union[int, None] = Field(
        ...,
        description="Column index of the cell in the table",
    )

    row_span: Union[int, None] = Field(
        ...,
        description="Number of rows spanned by the cell",
    )

    col_span: Union[int, None] = Field(
        ...,
        description="Number of columns spanned by the cell",
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
    id: Union[str, None] = Field(
        ...,
        description="Unique identifier of the key-value item",
    )
    key: Union[str, List[str]] = Field(..., description="Key cell id(s)")
    value: str = Field(..., description="Value cell id")


class TableGridCell(BaseSchema):
    row_keys: List[str] = Field(..., description="Row key cell id(s)")
    col_keys: List[str] = Field(..., description="Column key cell id(s)")
    col: int = Field(..., description="Column index of the cell in the grid")
    row: int = Field(..., description="Row index of the cell in the grid")
    value: str = Field(..., description="Value cell id(s)")

    # ---------- input: str | list[str] -> list[str] ----------
    @field_validator("row_keys", "col_keys", mode="before")
    @classmethod
    def _coerce_to_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return [x for x in v if isinstance(x, str) and x != ""]
        raise TypeError(f"Expected str or list[str], got {type(v)}")

    # ---------- output: list[str] -> str | list[str] ----------
    @field_serializer("row_keys", "col_keys")
    def _serialize_single_or_list(self, v: List[str]):
        if len(v) == 1:
            return v[0]
        return v


class TableGridSchema(BaseSchema):
    id: Union[str, None] = Field(
        ...,
        description="Unique identifier of the table grid",
    )
    box: conlist(int, min_length=4, max_length=4) = Field(
        ...,
        description="Bounding box of the table grid in the format [x1, y1, x2, y2]",
    )
    n_row: int = Field(
        ...,
        description="Number of rows in the table grid",
    )
    n_col: int = Field(
        ...,
        description="Number of columns in the table grid",
    )
    col_headers: List[List[str]] = Field(
        ...,
        description="2D array representing the column header cell ids",
    )
    data: List[List[Union[str, None]]] = Field(
        ...,
        description="2D array representing the table grid data with cell ids",
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

    def find_cell_by_id(self, cell_id: str) -> Optional[CellSchema]:
        """
        find for a cell by its ID.
        セルIDに対応するセルを返す

        Args:
            cell_id (str): 検索するセルID
        """
        return self.cells.get(str(cell_id))

    def search_cells_by_bbox(self, box: List[int]) -> List[CellSchema]:
        """
        search for a cell by its bounding box.
        セルの位置情報（bounding box）に対応するセルを返す

        Args:
            box (List[int]): 検索するバウンディングボックス [x1, y1, x2, y2]
        """
        cells = []
        for cell in self.cells.values():
            if cell.role == "group":
                continue

            if is_contained(box, cell.box, threshold=0.5):
                cells.append(cell)

        return cells

    def search_cells_below_key_text(self, key: str) -> List[CellSchema]:
        """
        search for cells located below the cells containing the given
        キー文字列を含むセルに隣接する下の位置にあるセルを返す

        Args:
            query (str): 検索するクエリ文字列. セルの内容に部分一致するものを検索
        """
        query_cells = self.search_cells_by_query(key)
        if query_cells is None:
            return []

        cells = []
        for cell in self.cells.values():
            if cell.role == "group":
                continue

            for query_cell in query_cells:
                if is_bottom_adjacent(query_cell.box, cell.box):
                    cells.append(cell)

        return cells

    def search_cells_right_of_key_text(self, key: str) -> List[CellSchema]:
        """
        search for cells located to the right of the cells containing the given
        キー文字列を含むセル隣接する右の位置にあるセルを返す

        Args:
            query (str): 検索するクエリ文字列. セルの内容に部分一致するものを検索
        """
        query_cells = self.search_cells_by_query(key)
        if query_cells is None:
            return []

        cells = []
        for cell in self.cells.values():
            if cell.role == "group":
                continue

            for query_cell in query_cells:
                if is_right_adjacent(query_cell.box, cell.box):
                    cells.append(cell)

        return cells

    def search_cells_left_of_key_text(self, key: str) -> List[CellSchema]:
        """
        search for cells located to the left of the cells containing the given
        キー文字列を含むセル隣接する左の位置にあるセルを返す

        Args:
            query (str): 検索するクエリ文字列. セルの内容に部分一致するものを検索
        """
        query_cells = self.search_cells_by_query(key)
        if query_cells is None:
            return []

        cells = []
        for cell in self.cells.values():
            if cell.role == "group":
                continue

            for query_cell in query_cells:
                if is_right_adjacent(cell.box, query_cell.box):
                    cells.append(cell)

        return cells

    def search_cells_upper_key_text(self, key: str) -> List[CellSchema]:
        """
        search for cells located above the cells containing the given
        キー文字列を含むセルに隣接する上の位置にあるセルを返す

        Args:
            query (str): 検索するクエリ文字列. セルの内容に部分一致するものを検索
        """
        query_cells = self.search_cells_by_query(key)
        if query_cells is None:
            return []

        cells = []
        for cell in self.cells.values():
            if cell.role == "group":
                continue

            for query_cell in query_cells:
                if is_bottom_adjacent(cell.box, query_cell.box):
                    cells.append(cell)

        return cells

    def search_cells_by_query(self, query: str) -> List[CellSchema]:
        """
        search for cells containing the query string in their contents.
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

    def search_kv_items_by_key(self, key: str) -> List[dict]:
        """
        search for key-value items or grid cells where the key matches the query string.
        クエリーに部分一致するキーを持つKVアイテムおよびグリッドセルを返す

        Args:
            key (str): 検索するクエリ文字列. キー部分に部分一致するものを検索
        """

        q = normalize(key)
        results: List[dict] = []

        # kv_items 側
        for kv_item in self.kv_items:
            key_cells = [self.cells.get(k) for k in kv_item.key]
            value_cell = self.cells.get(kv_item.value)

            key_text = "".join([(kc.contents or "") for kc in key_cells if kc])
            if q in normalize(key_text):
                results.append({"key": key_cells, "value": value_cell})

        # grids 側
        for grid in self.grids:
            for i, col in enumerate(grid.col_headers):
                col_cells = [self.cells.get(ck) for ck in col]
                col_text = "".join(
                    [self.safe_contents(c.contents) for c in col_cells if c]
                )

                if q in normalize(col_text):
                    value_cells = []
                    for row in grid.data:
                        cell_id = row[i]
                        value_cell = self.cells.get(cell_id)
                        results.append({"key": col_cells, "value": value_cells})

        return results

    def find_table_by_column_name(
        self,
        queries,
    ) -> "TableSemanticContentsSchema":
        """
        find for columns matching the specified column names.
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

        Args:
            grid (TableGridSchema): フィルタリング対象のテーブルグリッド
            queries (List[str]): 検索する列名のリスト
        """
        norm_queries = [normalize(q) for q in queries]
        result = {
            "id": grid.id,
            "data": [],
        }

        col_headers_filtered = defaultdict(int)
        for row in grid.data:
            filtered_row = []
            for i, cell in enumerate(row):
                has_query = False
                key_contents = [
                    self.cells.get(rk).contents or "" for rk in grid.col_headers[i]
                ]
                nk = normalize("".join(key_contents))
                for q in norm_queries:
                    if q in nk:
                        has_query = True
                        break
                if has_query:
                    filtered_row.append(cell)
                    col_headers_filtered[tuple(grid.col_headers[i])] += 1

            if filtered_row:
                result["data"].append(filtered_row)
                result["n_col"] = len(filtered_row)

        result["n_row"] = len(result["data"])
        result["col_headers"] = list(col_headers_filtered.keys())

        return TableGridSchema(**result) if result["data"] else None


class TableSemanticContentsExport:
    def __init__(self, table: TableSemanticContentsSchema):
        self.table = table

    def grids_to_dataframe(self, grid_id: str = "g0") -> DataFrame | None:
        grids = self.table.view.grids_to_dicts()
        for g in grids:
            if str(g["id"]) != str(grid_id):
                continue
            return pd.DataFrame(g["rows"])
        return None

    def grids_to_csv(self, out_path, columns=None, grid_id="g0"):
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
            table_contents = self.table.find_table_by_column_name(queries=columns)

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

    def visualize_kv_items(self, img) -> np.ndarray:
        """
        Visualize key-value items.
        キーと値のペアの内容を可視化
        """
        return kv_items_visualizer(img, self.table)

    def visualize_grids(self, img) -> np.ndarray:
        """
        Visualize table grids.
        テーブルグリッドの内容を可視化
        """
        return grids_visualizer(img, self.table)

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
            v = t.safe_contents(kv.value)
            keys.append(k)
            vals.append(v)

        keys = make_unique_all(keys)
        for k, v in zip(keys, vals):
            parsed["_".join(map(str, k))] = str(v)

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
            for row in grid.data:
                parsed_row = {}
                col_key_list, value_list = [], []

                for i, cell in enumerate(row):
                    ck = [t.safe_contents(i) for i in grid.col_headers[i]]
                    v = t.safe_contents(cell)
                    col_key_list.append(ck)
                    value_list.append(v)

                col_key_list = make_unique_all(col_key_list)
                for ck, v in zip(col_key_list, value_list):
                    parsed_row["_".join(map(str, ck))] = v

                row_record_list.append(parsed_row)
            results.append({"id": grid.id, "rows": row_record_list})

        return results


class CellTemplateSchema(BaseSchema):
    # match に使う情報（どちらか入っていれば良い）
    id: Optional[str] = Field(None, description="Cell id for matching")
    box: Optional[conlist(int, min_length=4, max_length=4)] = Field(
        None, description="Cell bbox for matching"
    )

    # 上書き対象（差分でOK）
    role: Optional[str] = Field(None, description="Role override")
    contents: Optional[str] = Field(None, description="Contents override")


class TableSemanticContentsTemplateSchema(BaseSchema):
    id: Optional[str] = Field(
        None, description="Unique identifier of the table (optional)"
    )
    style: Optional[str] = Field(None, description="Border style (optional)")

    # table matchingに使う（最小）
    box: conlist(int, min_length=4, max_length=4) = Field(
        ..., description="Bounding box [x1, y1, x2, y2]"
    )

    # テンプレ側は “差分” なので Dict[str, CellTemplateSchema] でOK
    # （キーはセルIDを想定。idが無いケースもあるので値側にも id を持たせてる）
    cells: Dict[str, CellTemplateSchema] = Field(
        default_factory=dict,
        description="Template cells keyed by cell_id (or arbitrary key)",
    )

    # これらもテンプレで差し替えるなら optional にしておく
    kv_items: Optional[List[KvItemSchema]] = Field(
        None, description="Optional KV items override"
    )
    grids: Optional[List[TableGridSchema]] = Field(
        None, description="Optional grids override"
    )


class TableSemanticParserTemplateSchema(BaseSchema):
    meta: TemplateMetaSchema = Field(
        ...,
        description="Metadata related to the table semantic parsing",
    )

    tables: List[TableSemanticContentsTemplateSchema] = Field(
        ...,
        description="List of tables with semantic information",
    )

    def find_table_by_id(
        self, table_id: str
    ) -> Union[TableSemanticContentsSchema, None]:
        """
        Search for a table by its ID.
        テーブルIDに対応するテーブルを返す

        Args:
            table_id (str): 検索するテーブルID
        """
        for table in self.tables:
            if table.id == str(table_id):
                return table


class TableSemanticParserSchema(BaseSchema):
    tables: List[TableSemanticContentsSchema] = Field(
        ...,
        description="List of tables with semantic information",
    )

    words: List[WordPrediction] = Field(
        ...,
        description="List of recognized words in the document",
    )

    def search_words_by_position(self, bbox) -> str:
        """
        Search for words by their bounding box.
        位置情報（bounding box）に対応する文字列を返す

        Args:
            box (List[int]): 検索するバウンディングボックス [x1, y1, x2, y2]
        """
        words = []
        for word in self.words:
            word_box = quad_to_xyxy(word.points)
            if is_contained(bbox, word_box, threshold=0.5):
                word = ParagraphSchema(
                    box=word_box,
                    contents=word.content,
                    direction=word.direction,
                    order=0,
                    role=None,
                )

                words.append(word)

        word_direction = [word.direction for word in words]
        cnt_horizontal = word_direction.count("horizontal")
        cnt_vertical = word_direction.count("vertical")

        element_direction = (
            "horizontal" if cnt_horizontal > cnt_vertical else "vertical"
        )
        order = "left2right" if element_direction == "horizontal" else "right2left"
        words = prediction_reading_order(words, order)
        words = sorted(words, key=lambda x: x.order)

        return "".join([word.contents for word in words])

    def load_json(self, json_path: str) -> "TableSemanticParserSchema":
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return TableSemanticParserSchema.model_validate(data)

    def find_table_by_id(
        self, table_id: str
    ) -> Union[TableSemanticContentsSchema, None]:
        """
        Search for a table by its ID.
        テーブルIDに対応するテーブルを返す

        Args:
            table_id (str): 検索するテーブルID
        """
        for table in self.tables:
            if table.id == str(table_id):
                return table

    def find_table_by_position(
        self, box: List[int]
    ) -> Union[TableSemanticContentsSchema, None]:
        """
        Search for a table by its bounding box.
        テーブルの位置情報（bounding box）に対応するテーブルを返す

        Args:
            box (List[int]): 検索するバウンディングボックス [x1, y1, x2, y2]
        """
        ratios = []
        for table in self.tables:
            overlap_ratio = calc_overlap_ratio(box, table.box)[0]
            ratios.append(overlap_ratio)

        if not ratios:
            return None

        max_idx = ratios.index(max(ratios))
        return self.tables[max_idx] if ratios[max_idx] > 0.5 else None

    def search_kv_items_by_key(self, key: str) -> List[dict]:
        """
        search for key-value items or grid cells where the key matches the query string.
        クエリーに部分一致するキーを持つKVアイテムおよびグリッドセルを返す

        Args:
            key (str): 検索するクエリ文字列. キー部分に部分一致するものを検索
        """

        results: List[dict] = []
        for table in self.tables:
            table_results = table.search_kv_items_by_key(key)
            results.extend(table_results)

        return results

    def load_template_json(self, template_path: str) -> "TableSemanticParserSchema":
        with open(template_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        template = TableSemanticParserTemplateSchema.model_validate(data)
        return apply_table_template(self, template)

    def save_template_json(
        self, out_path: str, include_kv: bool = True, include_grids: bool = True
    ):
        template_tables: List[TableSemanticContentsTemplateSchema] = []

        for t in self.tables:
            tmp_cells: Dict[str, CellTemplateSchema] = {}
            for cid, c in t.cells.items():
                if c.role == "group":
                    continue

                tmp_cells[str(cid)] = CellTemplateSchema(
                    id=str(c.id) if c.id is not None else str(cid),
                    box=list(c.box) if c.box is not None else None,
                    role=c.role,
                    contents=c.contents,
                )

            template_tables.append(
                TableSemanticContentsTemplateSchema(
                    id=t.id,
                    style=t.style,
                    box=list(t.box),
                    cells=tmp_cells,
                    kv_items=t.kv_items if include_kv else None,
                    grids=t.grids if include_grids else None,
                )
            )

        template = TableSemanticParserTemplateSchema(
            meta=TemplateMetaSchema(),
            tables=template_tables,
        )

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                template.model_dump(exclude_none=True), f, ensure_ascii=False, indent=4
            )


def _match_cell(
    table: TableSemanticContentsSchema,
    tcell: CellTemplateSchema,
    policy: str = "cell_id",
) -> CellSchema | None:
    if policy == "cell_id":
        if not tcell.id:
            return None
        return table.cells.get(str(tcell.id))

    if policy == "bbox":
        if not tcell.box:
            return None
        candidates = table.search_cells_by_bbox(list(tcell.box))
        return candidates[0] if candidates else None

    return None


def apply_table_template(
    tables: TableSemanticParserSchema,
    tmpl: TableSemanticParserTemplateSchema,
) -> TableSemanticParserSchema:
    policy = getattr(tmpl.meta, "match_policy", "cell_id")

    for tmp_table in tmpl.tables:
        table = tables.find_table_by_position(tmp_table.box)
        if table is None:
            continue

        # cells: role/contents をテンプレ優先で上書き
        for tcell in tmp_table.cells.values():
            cell = _match_cell(table, tcell, policy=policy)
            if cell is None:
                continue
            if tcell.role is not None:
                cell.role = tcell.role
            if tcell.contents is not None:
                cell.contents = tcell.contents

        # kv_items / grids: テンプレが持っているなら差し替え
        if tmp_table.kv_items is not None:
            table.kv_items = tmp_table.kv_items
        if tmp_table.grids is not None:
            table.grids = tmp_table.grids

    return tables
