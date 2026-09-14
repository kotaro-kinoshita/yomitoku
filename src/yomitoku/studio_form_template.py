"""Apply YomiToku Studio form-template JSON to OCR results.

Studio templates own the table/cell structure. Applying one therefore only
needs OCR; table, cell, and semantic inference must not be run again.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import cv2
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .ocr import OCR
from .reading_order import prediction_reading_order
from .schemas.document_analyzer import Element, ParagraphSchema, WordPrediction
from .schemas.table_semantic_parser import (
    CellSchema,
    KvItemSchema,
    TableGridSchema,
    TableSemanticContentsSchema,
    TableSemanticParserSchema,
)
from .utils.misc import calc_overlap_ratio, quad_to_xyxy


class _StudioModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class StudioTemplateCell(_StudioModel):
    id: str
    norm_box: list[float] = Field(alias="normBox", min_length=4, max_length=4)
    role: str | None
    row: int | None = None
    col: int | None = None
    row_span: int | None = Field(None, alias="rowSpan")
    col_span: int | None = Field(None, alias="colSpan")
    meta: dict[str, Any] = Field(default_factory=dict)
    contents: str = ""

    @field_validator("norm_box")
    @classmethod
    def validate_box(cls, value: list[float]) -> list[float]:
        if not all(math.isfinite(n) for n in value):
            raise ValueError("normBox must contain finite numbers")
        if value[0] > value[2] or value[1] > value[3]:
            raise ValueError("normBox coordinates are reversed")
        return value


class StudioTemplateKvItem(_StudioModel):
    id: str | None = None
    key: list[str]
    value: str


class StudioTemplateGrid(_StudioModel):
    id: str | None = None
    norm_box: list[float] = Field(alias="normBox", min_length=4, max_length=4)
    n_row: int = Field(alias="nRow")
    n_col: int = Field(alias="nCol")
    col_headers: list[list[str]] = Field(alias="colHeaders")
    data: list[list[str | None]]


class StudioTemplateTable(_StudioModel):
    id: str
    norm_box: list[float] = Field(alias="normBox", min_length=4, max_length=4)
    style: str = "border"
    cells: list[StudioTemplateCell]
    kv_items: list[StudioTemplateKvItem] = Field(default_factory=list, alias="kvItems")
    grids: list[StudioTemplateGrid] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self):
        cell_ids = [cell.id for cell in self.cells]
        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError(f"table {self.id!r} contains duplicate cell ids")
        known = set(cell_ids)

        for item in self.kv_items:
            unknown = [ref for ref in [*item.key, item.value] if ref not in known]
            if unknown:
                raise ValueError(
                    f"table {self.id!r} kv item references unknown cells: {unknown}"
                )
        for grid in self.grids:
            refs = [ref for header in grid.col_headers for ref in header]
            refs.extend(ref for row in grid.data for ref in row if ref is not None)
            unknown = [ref for ref in refs if ref not in known]
            if unknown:
                raise ValueError(
                    f"table {self.id!r} grid references unknown cells: {unknown}"
                )
        return self


class StudioTemplateParagraph(_StudioModel):
    id: str
    norm_box: list[float] = Field(alias="normBox", min_length=4, max_length=4)
    role: str | None = None
    contents: str = ""


class StudioTemplateStructure(_StudioModel):
    tables: list[StudioTemplateTable]
    paragraphs: list[StudioTemplateParagraph] = Field(default_factory=list)


class StudioFormTemplate(_StudioModel):
    kind: Literal["form-template"]
    version: Literal[2, 3]
    structure: StudioTemplateStructure
    fields: list[dict[str, Any]]

    @model_validator(mode="after")
    def validate_table_ids(self):
        ids = [table.id for table in self.structure.tables]
        if len(ids) != len(set(ids)):
            raise ValueError("template contains duplicate table ids")
        return self


def load_studio_form_template(path: str | Path) -> StudioFormTemplate:
    """Load and validate a Studio v2/v3 form-template JSON file."""
    with open(path, "r", encoding="utf-8") as stream:
        return StudioFormTemplate.model_validate(json.load(stream))


def _denormalize(box: list[float], width: int, height: int) -> list[int]:
    return [
        int(round(box[0] * width)),
        int(round(box[1] * height)),
        int(round(box[2] * width)),
        int(round(box[3] * height)),
    ]


def _ordered_text(words: list[WordPrediction], separator: str = "") -> str:
    if not words:
        return ""
    elements = [
        ParagraphSchema(
            box=list(quad_to_xyxy(word.points)),
            contents=word.content,
            direction=word.direction,
            order=0,
            role=None,
        )
        for word in words
    ]
    horizontal = sum(word.direction == "horizontal" for word in words)
    direction = "left2right" if horizontal >= len(words) - horizontal else "right2left"
    prediction_reading_order(elements, direction)
    elements.sort(key=lambda element: element.order)
    return separator.join((element.contents or "") for element in elements).strip()


def apply_studio_form_template(
    template: StudioFormTemplate,
    words: list[WordPrediction],
    width: int,
    height: int,
) -> TableSemanticParserSchema:
    """Rebuild table semantics from a Studio template and current-page OCR words."""
    tables: list[TableSemanticContentsSchema] = []

    for source_table in template.structure.tables:
        cells = {
            source.id: CellSchema(
                id=source.id,
                box=_denormalize(source.norm_box, width, height),
                role=source.role,
                contents="",
                row=source.row,
                col=source.col,
                row_span=source.row_span,
                col_span=source.col_span,
                meta={**source.meta, "fromTemplate": True},
            )
            for source in source_table.cells
        }

        assigned: dict[str, list[WordPrediction]] = defaultdict(list)
        for word in words:
            word_box = quad_to_xyxy(word.points)
            best_id = None
            best_ratio = 0.0
            for cell_id, cell in cells.items():
                if cell.role == "group":
                    continue
                ratio, _ = calc_overlap_ratio(cell.box, word_box)
                if ratio > best_ratio:
                    best_id, best_ratio = cell_id, ratio
            if best_id is not None and best_ratio >= 0.2:
                assigned[best_id].append(word)

        source_cells = {cell.id: cell for cell in source_table.cells}
        for cell_id, cell in cells.items():
            text = _ordered_text(assigned[cell_id])
            source = source_cells[cell_id]
            # Never copy a template value into an unread/empty value cell.
            cell.contents = (
                text
                if text
                else source.contents
                if cell.role in ("header", "group")
                else ""
            )

        tables.append(
            TableSemanticContentsSchema(
                id=source_table.id,
                style=source_table.style,
                box=_denormalize(source_table.norm_box, width, height),
                cells=cells,
                kv_items=[
                    KvItemSchema(
                        id=item.id,
                        key=item.key,
                        value=item.value,
                        box=None,
                    )
                    for item in source_table.kv_items
                ],
                grids=[
                    TableGridSchema(
                        id=grid.id,
                        box=_denormalize(grid.norm_box, width, height),
                        n_row=grid.n_row,
                        n_col=grid.n_col,
                        col_headers=grid.col_headers,
                        data=grid.data,
                    )
                    for grid in source_table.grids
                ],
            )
        )

    paragraphs: list[Element] = []
    for source in template.structure.paragraphs:
        box = _denormalize(source.norm_box, width, height)
        inside = []
        for word in words:
            ratio, _ = calc_overlap_ratio(box, quad_to_xyxy(word.points))
            if ratio >= 0.5:
                inside.append(word)
        paragraphs.append(
            Element(
                id=source.id,
                box=box,
                score=1.0,
                role=source.role,
                contents=_ordered_text(inside, separator="\n"),
            )
        )

    return TableSemanticParserSchema(tables=tables, paragraphs=paragraphs, words=words)


class StudioFormTemplateParser:
    """OCR-only parser used by ``yomitoku_table --studio-template``."""

    def __init__(
        self,
        template_path: str | Path,
        *,
        configs: dict,
        device: str,
        visualize: bool,
    ):
        self.template = load_studio_form_template(template_path)
        self.visualize = visualize
        self.ocr = OCR(configs=configs, device=device, visualize=visualize)

    def __call__(self, image):
        ocr_result, vis_ocr = self.ocr(image)
        height, width = image.shape[:2]
        result = apply_studio_form_template(
            self.template, ocr_result.words, width=width, height=height
        )

        vis_layout = image.copy()
        if self.visualize:
            for table in result.tables:
                cv2.rectangle(
                    vis_layout,
                    (table.box[0], table.box[1]),
                    (table.box[2], table.box[3]),
                    (255, 0, 0),
                    3,
                )
                for cell in table.cells.values():
                    cv2.rectangle(
                        vis_layout,
                        (cell.box[0], cell.box[1]),
                        (cell.box[2], cell.box[3]),
                        (0, 180, 0),
                        1,
                    )
        return result, vis_layout, vis_ocr if vis_ocr is not None else image.copy()
