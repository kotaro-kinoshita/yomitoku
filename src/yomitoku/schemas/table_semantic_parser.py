import re
import json
import os

from collections import defaultdict
from typing import List, Union, Optional, Literal, Any
from pydantic import conlist, Field, PrivateAttr
from typing import Dict

from ..base import BaseSchema
from . import WordPrediction, ParagraphSchema, Element
from ..reading_order import prediction_reading_order


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


class TableDetectorSchema(BaseSchema):
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
    box: Union[conlist(int, min_length=4, max_length=4), None] = Field(
        None,
        description="Bounding box of the key-value item in the format [x1, y1, x2, y2]",
    )


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

    def safe_contents(self, cell_id: str, ignore_space=True) -> str:
        c = self.cells.get(cell_id)

        contents = c.contents or "" if c is not None else ""
        if not ignore_space:
            return contents

        return contents.replace(" ", "")

    def find_cell_by_id(self, cell_id: str) -> Optional[CellSchema]:
        return self.cells.get(str(cell_id))

    def search_cells_by_bbox(self, box: List[int]) -> List[CellSchema]:
        cells = []
        for cell in self.cells.values():
            if cell.role == "group":
                continue

            if is_contained(box, cell.box, threshold=0.5):
                cells.append(cell)

        return cells

    def search_cells_below_key_text(self, key: str) -> List[CellSchema]:
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
        result["box"] = grid.box

        return TableGridSchema(**result) if result["data"] else None


class TableSemanticContentsExport:
    def __init__(self, table: TableSemanticContentsSchema):
        self.table = table

    def to_json(self, out_path) -> str:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        grids = self.table.view.grids_to_dict()
        kv_items = self.table.view.kv_items_to_dict()

        table_contents = {
            "kv_items": kv_items,
            "grids": grids,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(table_contents, f, ensure_ascii=False, indent=4)

    def grids_to_csv(
        self, out_path, columns=None, ignore_space=True
    ) -> List[List[str]]:
        table_contents = self.table
        if columns is not None:
            table_contents = self.table.find_table_by_column_name(queries=columns)

        dirname = os.path.dirname(out_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        csvs = []
        for table_grid in table_contents.grids:
            csv = []
            for row in table_grid.data:
                row_data = []
                for cell_id in row:
                    cell_text = table_contents.safe_contents(cell_id, ignore_space)
                    row_data.append(cell_text)
                csv.append(row_data)

            basename = out_path.rsplit(".", 1)[0]
            out_path = f"{basename}_{table_grid.id}.csv"

            with open(out_path, "w", encoding="utf-8") as f:
                for row in csv:
                    f.write(",".join(row) + "\n")

            csvs.append(csv)

        return csvs

    def grids_to_json(self, out_path) -> str:
        grids = self.table.view.grids_to_dict()

        dirname = os.path.dirname(out_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(grids, f, ensure_ascii=False, indent=4)

        return grids

    def kv_items_to_json(self, out_path) -> str:
        kv_items = self.table.view.kv_items_to_dict()

        dirname = os.path.dirname(out_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(kv_items, f, ensure_ascii=False, indent=4)

        return kv_items


class TableSemanticContentsView:
    def __init__(self, table: TableSemanticContentsSchema):
        self.table = table

    def kv_items_to_dict(self) -> dict:
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

    def grids_to_dict(self, ignore_space=True) -> list[dict]:
        t = self.table
        results = []
        for grid in t.grids:
            row_record_list = []
            for row in grid.data:
                parsed_row = {}

                cell_id_list = set()
                col_key_list, value_list = [], []

                for i, cell in enumerate(row):
                    if cell in grid.col_headers[i]:
                        continue

                    ck = [t.safe_contents(i, ignore_space) for i in grid.col_headers[i]]
                    v = t.safe_contents(cell, ignore_space)

                    if cell in cell_id_list:
                        continue

                    col_key_list.append(ck)
                    value_list.append(v)
                    cell_id_list.add(cell)

                col_key_list = make_unique_all(col_key_list)
                for ck, v in zip(col_key_list, value_list):
                    parsed_row["_".join(map(str, ck))] = v

                if parsed_row:
                    row_record_list.append(parsed_row)
            results.append({"id": grid.id, "rows": row_record_list})

        return results


class CellTemplateSchema(BaseSchema):
    id: Optional[str] = Field(None, description="Cell id for matching")
    box: Optional[conlist(int, min_length=4, max_length=4)] = Field(
        None, description="Cell bbox for matching"
    )

    role: Optional[str] = Field(None, description="Role override")
    contents: Optional[str] = Field(None, description="Contents override")


class TableSemanticContentsTemplateSchema(BaseSchema):
    id: Optional[str] = Field(
        None, description="Unique identifier of the table (optional)"
    )
    style: Optional[str] = Field(None, description="Border style (optional)")

    box: conlist(int, min_length=4, max_length=4) = Field(
        ..., description="Bounding box [x1, y1, x2, y2]"
    )

    cells: Dict[str, CellTemplateSchema] = Field(
        default_factory=dict,
        description="Template cells keyed by cell_id (or arbitrary key)",
    )

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
        for table in self.tables:
            if table.id == str(table_id):
                return table


class TableSemanticParserSchema(BaseSchema):
    tables: List[TableSemanticContentsSchema] = Field(
        ...,
        description="List of tables with semantic information",
    )

    paragraphs: List[Element] = Field(
        ...,
        description="List of recognized paragraphs in the document",
    )

    words: List[WordPrediction] = Field(
        ...,
        description="List of recognized words in the document",
    )

    def search_words_by_position(self, bbox) -> str:
        words = []
        for word in self.words:
            word_box = quad_to_xyxy(word.points)
            if is_contained(bbox, word_box, threshold=0.5):
                word = ParagraphSchema(
                    box=word_box,
                    contents=word.content,
                    direction=word.direction,
                    role=None,
                    order=None,
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

    @classmethod
    def load_json(self, json_path: str) -> "TableSemanticParserSchema":
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return TableSemanticParserSchema.model_validate(data)

    def to_csv(self, outdir):
        for table in self.tables:
            table.export.grids_to_csv(
                out_path=f"{outdir}/table_{table.id}.csv",
            )

    def to_dict(self):
        results = {}
        for table in self.tables:
            result = {
                "kv_items": table.view.kv_items_to_dict(),
                "grids": table.view.grids_to_dict(),
            }
            results[table.id] = result

        return results

    def find_table_by_id(
        self, table_id: str
    ) -> Union[TableSemanticContentsSchema, None]:
        for table in self.tables:
            if table.id == str(table_id):
                return table

    def find_table_by_position(
        self, box: List[int]
    ) -> Union[TableSemanticContentsSchema, None]:
        ratios = []
        for table in self.tables:
            overlap_ratio = calc_overlap_ratio(box, table.box)[0]
            ratios.append(overlap_ratio)

        if not ratios:
            return None

        max_idx = ratios.index(max(ratios))
        return self.tables[max_idx] if ratios[max_idx] > 0.5 else None

    def search_kv_items_by_key(self, key: str) -> List[dict]:
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
