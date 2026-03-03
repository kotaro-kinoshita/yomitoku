import json
import os
import re
from typing import Any, Dict, List

import numpy as np

from ..schemas.table_semantic_parser import TableSemanticParserSchema
from ..utils.logger import set_logger
from ..utils.misc import calc_overlap_ratio, quad_to_xyxy
from .pipeline import _build_output, _normalize_resolved_fields
from .resolver import ResolvedElement, ResolvedField
from .schema import ExtractionSchema
from .visualizer import extraction_visualizer

logger = set_logger(__name__, "INFO")


def _normalize_text(text: str) -> str:
    return re.sub(r"[ \u3000]", "", text)


def _extract_scalar_by_cell_id(semantic_info, field_schema):
    for table in semantic_info.tables:
        cell = table.find_cell_by_id(field_schema.cell_id)
        if cell is not None:
            contents = cell.contents or ""
            return ResolvedField(
                name=field_schema.name,
                value=contents,
                raw_text=contents,
                elements=[
                    ResolvedElement(
                        id=cell.id,
                        box=list(cell.box),
                        contents=contents,
                    )
                ],
                confidence="high",
                source="cell_id",
            )
    return None


def _extract_scalar_by_bbox(semantic_info, field_schema):
    for table in semantic_info.tables:
        cells = table.search_cells_by_bbox(field_schema.bbox)
        if cells:
            cell = cells[0]
            contents = cell.contents or ""
            return ResolvedField(
                name=field_schema.name,
                value=contents,
                raw_text=contents,
                elements=[
                    ResolvedElement(
                        id=cell.id,
                        box=list(cell.box),
                        contents=contents,
                    )
                ],
                confidence="high",
                source="bbox",
            )
    return None


def _extract_scalar_by_regex(semantic_info, field_schema):
    pattern = re.compile(field_schema.regex)

    for table in semantic_info.tables:
        for cell_id, cell in table.cells.items():
            if cell.role == "group" or not cell.contents:
                continue
            m = pattern.search(cell.contents)
            if m:
                matched = m.group(0)
                return ResolvedField(
                    name=field_schema.name,
                    value=matched,
                    raw_text=cell.contents,
                    elements=[
                        ResolvedElement(
                            id=cell.id,
                            box=list(cell.box),
                            contents=cell.contents,
                        )
                    ],
                    confidence="high",
                    source="regex",
                )

    for i, paragraph in enumerate(semantic_info.paragraphs):
        if not paragraph.contents:
            continue
        m = pattern.search(paragraph.contents)
        if m:
            matched = m.group(0)
            pid = paragraph.id or f"p{i}"
            return ResolvedField(
                name=field_schema.name,
                value=matched,
                raw_text=paragraph.contents,
                elements=[
                    ResolvedElement(
                        id=pid,
                        box=list(paragraph.box),
                        contents=paragraph.contents,
                    )
                ],
                confidence="high",
                source="regex",
            )

    for i, word in enumerate(semantic_info.words):
        if not word.content:
            continue
        m = pattern.search(word.content)
        if m:
            matched = m.group(0)
            wid = f"w{i}"
            box = quad_to_xyxy(word.points)
            return ResolvedField(
                name=field_schema.name,
                value=matched,
                raw_text=word.content,
                elements=[
                    ResolvedElement(
                        id=wid,
                        box=box,
                        contents=word.content,
                    )
                ],
                confidence="high",
                source="regex",
            )

    return None


def _extract_scalar_field(semantic_info, field_schema):
    if field_schema.cell_id:
        result = _extract_scalar_by_cell_id(semantic_info, field_schema)
        if result is not None:
            return result

    if field_schema.bbox:
        result = _extract_scalar_by_bbox(semantic_info, field_schema)
        if result is not None:
            return result

    if field_schema.description:
        kv_results = semantic_info.search_kv_items_by_key(field_schema.description)
        if kv_results:
            kv = kv_results[0]
            value_cell = kv["value"]
            if value_cell is not None:
                contents = value_cell.contents or ""
                return ResolvedField(
                    name=field_schema.name,
                    value=contents,
                    raw_text=contents,
                    elements=[
                        ResolvedElement(
                            id=value_cell.id,
                            box=list(value_cell.box),
                            contents=contents,
                        )
                    ],
                    confidence="high",
                    source="kv",
                )

    if field_schema.description:
        for table in semantic_info.tables:
            cells = table.search_cells_by_query(field_schema.description)
            if cells:
                cell = cells[0]
                contents = cell.contents or ""
                return ResolvedField(
                    name=field_schema.name,
                    value=contents,
                    raw_text=contents,
                    elements=[
                        ResolvedElement(
                            id=cell.id,
                            box=list(cell.box),
                            contents=contents,
                        )
                    ],
                    confidence="medium",
                    source="cell_query",
                )

    if field_schema.description:
        q = _normalize_text(field_schema.description)
        for paragraph in semantic_info.paragraphs:
            if paragraph.contents and q in _normalize_text(paragraph.contents):
                contents = paragraph.contents or ""
                pid = paragraph.id or "p0"
                return ResolvedField(
                    name=field_schema.name,
                    value=contents,
                    raw_text=contents,
                    elements=[
                        ResolvedElement(
                            id=pid,
                            box=list(paragraph.box),
                            contents=contents,
                        )
                    ],
                    confidence="medium",
                    source="paragraph",
                )

    if field_schema.regex:
        result = _extract_scalar_by_regex(semantic_info, field_schema)
        if result is not None:
            return result

    return ResolvedField(
        name=field_schema.name,
        value="",
        raw_text="",
        elements=[],
        confidence="low",
        source="not_found",
    )


def _match_col_header(table, header_cell_ids, col_schema):
    if col_schema.cell_id:
        return col_schema.cell_id in header_cell_ids

    if col_schema.bbox:
        for cid in header_cell_ids:
            cell = table.cells.get(cid)
            if cell and cell.box:
                overlap = calc_overlap_ratio(col_schema.bbox, list(cell.box))[0]
                if overlap > 0.5:
                    return True
        return False

    match_key = col_schema.description if col_schema.description else col_schema.name
    header_text = "".join(
        (table.cells.get(cid).contents or "")
        for cid in header_cell_ids
        if table.cells.get(cid)
    )
    return _normalize_text(match_key) in _normalize_text(header_text)


def _extract_table_field(semantic_info, field_schema):
    if not field_schema.columns:
        return ResolvedField(
            name=field_schema.name,
            value=[],
            raw_text="",
            elements=[],
            confidence="low",
            source="not_found",
        )

    records = []
    elements = []

    for table in semantic_info.tables:
        for grid in table.grids:
            col_index_map: Dict[str, List[int]] = {}
            for col_idx, header_cell_ids in enumerate(grid.col_headers):
                for col_schema in field_schema.columns:
                    if _match_col_header(table, header_cell_ids, col_schema):
                        col_index_map.setdefault(col_schema.name, []).append(col_idx)
                        break

            if not col_index_map:
                continue

            all_header_ids = {cid for col_ids in grid.col_headers for cid in col_ids}

            for row in grid.data:
                non_none_ids = [cid for cid in row if cid is not None]
                if non_none_ids and all(cid in all_header_ids for cid in non_none_ids):
                    continue

                row_record = {}
                for schema_col_name, col_indices in col_index_map.items():
                    values = []
                    cell_ids = []
                    for col_idx in col_indices:
                        if col_idx >= len(row):
                            continue
                        cell_id = row[col_idx]
                        if cell_id is not None:
                            cell = table.cells.get(cell_id)
                            if cell:
                                contents = cell.contents or ""
                                values.append(contents)
                                cell_ids.append(cell_id)
                                elements.append(
                                    ResolvedElement(
                                        id=cell_id,
                                        box=list(cell.box),
                                        contents=contents,
                                        label=schema_col_name,
                                    )
                                )
                            else:
                                cell_ids.append(cell_id)
                    row_record[schema_col_name] = {
                        "value": "".join(values),
                        "cell_ids": cell_ids,
                    }
                if row_record:
                    records.append(row_record)

    confidence = "high" if records else "low"
    source = "grid" if records else "not_found"

    return ResolvedField(
        name=field_schema.name,
        value=records,
        raw_text="",
        elements=elements,
        confidence=confidence,
        source=source,
    )


def run_rule_extraction(
    semantic_info: TableSemanticParserSchema,
    img: np.ndarray,
    schema: ExtractionSchema,
    no_normalize: bool = False,
    visualize: bool = False,
    simple: bool = False,
    outdir: str = "results",
    filename: str = "output",
) -> Dict[str, Any]:
    resolved: List[ResolvedField] = []

    for field_schema in schema.fields:
        if field_schema.structure == "table":
            rf = _extract_table_field(semantic_info, field_schema)
        else:
            rf = _extract_scalar_field(semantic_info, field_schema)
        resolved.append(rf)

    resolved = _normalize_resolved_fields(resolved, schema, skip_normalize=no_normalize)

    from .pipeline import _build_simple_output

    output = _build_simple_output(resolved) if simple else _build_output(resolved)

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{filename}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"Output JSON: {out_path}")

    if visualize:
        vis_img = extraction_visualizer(img, resolved)
        vis_path = os.path.join(outdir, f"{filename}_extract_vis.jpg")
        from ..utils.misc import save_image

        save_image(vis_img, vis_path)
        logger.info(f"Visualization: {vis_path}")

    return output
