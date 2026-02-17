import json
from typing import Dict, List, Optional

from ..schemas.table_semantic_parser import (
    TableSemanticContentsSchema,
    TableSemanticParserSchema,
)
from ..utils.misc import quad_to_xyxy
from .schema import ExtractionSchema

SYSTEM_PROMPT = """\
You are a document data extraction assistant.
You receive OCR-analyzed document data and extract information according to a schema.
Respond with valid JSON only. No explanation outside JSON.

CRITICAL RULES:
1. Return a JSON object with a single key "results" containing an array.
2. The array MUST have exactly one entry per schema field, in the same order as the schema.
3. Each entry MUST have these exact keys: "name", "value", "raw_text", "cell_ids", "confidence", "source".
4. For structure=kv fields:
   - "value" MUST be a plain string (e.g. "東京都渋谷区"). NEVER an object or array.
   - "cell_ids" is a list of IDs where the value was found (e.g. ["c5"] or ["p0"] or ["w3"]).
5. For structure=table fields:
   - "value" MUST be a list of row objects. Each row has column names as keys.
   - Each column value is an object with "value" (string) and "cell_ids" (list).
6. "confidence": "high", "medium", or "low".
7. "source": one of "kv", "grid", "cell_search", "paragraph", "word".
8. If a value cannot be found, return "value": "", "cell_ids": [], "confidence": "low", "source": "not_found".
9. Cell IDs are listed in brackets [id1,id2,...] at the end of each entry. Use these IDs directly in your response.
10. For structure=table: each column entry shows "header: value [id1,id2,...]". Use the value text as "value" and the bracketed IDs as "cell_ids".
"""


def _build_paragraphs_section(
    semantic_info: TableSemanticParserSchema,
) -> str:
    lines = []
    for i, p in enumerate(semantic_info.paragraphs):
        pid = p.id if p.id else f"p{i}"
        contents = (p.contents or "").replace("\n", " ")
        role = p.role or ""
        box = list(p.box)
        lines.append(f"  {pid}: contents={contents!r}, role={role!r}, box={box}")
    return "\n".join(lines)


def _build_tables_section(
    tables: List[TableSemanticContentsSchema],
    table_id_filter: Optional[str] = None,
) -> str:
    sections = []
    for table in tables:
        if table_id_filter and table.id != table_id_filter:
            continue

        parts = [f"--- Table {table.id} (style={table.style}) ---"]
        referenced_ids: set = set()

        # KV Items: text and IDs separated
        if table.kv_items:
            parts.append("KV Items:")
            for kv in table.kv_items:
                key_ids = kv.key if isinstance(kv.key, list) else [kv.key]
                key_texts = []
                all_ids = []
                for kid in key_ids:
                    key_texts.append(_safe_contents(table, kid))
                    all_ids.append(kid)
                    referenced_ids.add(kid)
                key_str = "".join(key_texts)

                val_text = _safe_contents(table, kv.value)
                all_ids.append(kv.value)
                referenced_ids.add(kv.value)
                id_list = ",".join(all_ids)
                parts.append(f"  - {key_str}: {val_text} [{id_list}]")

        # Grids: text and IDs separated
        for grid in table.grids:
            grid_id = grid.id or "g?"
            parts.append(f"Grid {grid_id}:")

            # Build header text and IDs per column
            col_header_texts = []
            col_header_ids = []
            for col_ids in grid.col_headers:
                texts = []
                ids = []
                for cid in col_ids:
                    texts.append(_safe_contents(table, cid))
                    ids.append(cid)
                    referenced_ids.add(cid)
                col_header_texts.append("".join(texts))
                col_header_ids.append(ids)

            # Data rows: merge columns sharing the same header text
            for row_idx, row in enumerate(grid.data):
                merged = _merge_row_by_header(
                    row, col_header_texts, col_header_ids, table, referenced_ids
                )
                row_parts = []
                for header_text, _h_ids, val_texts, val_ids in merged:
                    val_str = "".join(val_texts)
                    id_list = ",".join(val_ids)
                    row_parts.append(f"{header_text}: {val_str} [{id_list}]")
                parts.append(f"  Row {row_idx}: {' | '.join(row_parts)}")

        # Unassigned cells (not referenced by any KV/Grid, not group)
        unassigned = []
        for cell_id, cell in table.cells.items():
            if cell.role == "group":
                continue
            if cell_id not in referenced_ids:
                contents = (cell.contents or "").replace("\n", " ")
                unassigned.append(
                    f"  {cell_id}: contents={contents!r}, role={cell.role!r}"
                )
        if unassigned:
            parts.append("Unassigned Cells:")
            parts.extend(unassigned)

        sections.append("\n".join(parts))

    return "\n\n".join(sections)


def _safe_contents(table, cell_id: str) -> str:
    """Get cell contents safely, returning empty string if not found."""
    cell = table.cells.get(cell_id)
    if cell is None:
        return ""
    return (cell.contents or "").replace("\n", " ")


def _merge_row_by_header(row, col_header_texts, col_header_ids, table, referenced_ids):
    """Merge columns with the same header text within a single row.

    Returns list of (header_text, header_cell_ids, value_texts, value_cell_ids) tuples.
    Columns sharing the same header text are merged into one entry.
    """
    from collections import OrderedDict

    merged = OrderedDict()
    for col_idx, cid in enumerate(row):
        header_text = (
            col_header_texts[col_idx] if col_idx < len(col_header_texts) else "?"
        )
        h_ids = col_header_ids[col_idx] if col_idx < len(col_header_ids) else []

        if header_text not in merged:
            merged[header_text] = {
                "h_ids": list(h_ids),
                "val_texts": [],
                "val_ids": [],
            }
        else:
            for hid in h_ids:
                if hid not in merged[header_text]["h_ids"]:
                    merged[header_text]["h_ids"].append(hid)

        if cid is not None:
            text = _safe_contents(table, cid)
            merged[header_text]["val_texts"].append(text)
            merged[header_text]["val_ids"].append(cid)
            referenced_ids.add(cid)

    return [
        (header_text, entry["h_ids"], entry["val_texts"], entry["val_ids"])
        for header_text, entry in merged.items()
    ]


def _build_words_section(
    semantic_info: TableSemanticParserSchema,
) -> str:
    lines = []
    for i, w in enumerate(semantic_info.words):
        wid = f"w{i}"
        content = (w.content or "").replace("\n", " ")
        box = quad_to_xyxy(w.points)
        lines.append(f"  {wid}: content={content!r}, box={box}")
    return "\n".join(lines)


def _build_schema_section(schema: ExtractionSchema) -> str:
    lines = ["Fields to extract:"]
    for field in schema.fields:
        desc = field.description or field.name
        if field.structure == "table":
            col_info = ""
            if field.columns:
                cols = [f"{c.name}(type={c.type})" for c in field.columns]
                col_info = f", columns=[{', '.join(cols)}]"
            lines.append(f"  - {field.name}: {desc} (structure=table{col_info})")
        else:
            lines.append(f"  - {field.name}: {desc} (structure=kv, type={field.type})")
    return "\n".join(lines)


def _build_response_format(schema: ExtractionSchema) -> str:
    results_example = []
    for field in schema.fields:
        if field.structure == "table" and field.columns:
            row_example = {}
            for col in field.columns:
                row_example[col.name] = {"value": "...", "cell_ids": ["c0"]}
            entry = {
                "name": field.name,
                "value": [row_example],
                "raw_text": "",
                "cell_ids": [],
                "confidence": "high",
                "source": "grid",
            }
        else:
            entry = {
                "name": field.name,
                "value": "extracted text here",
                "raw_text": "original text here",
                "cell_ids": ["c0"],
                "confidence": "high",
                "source": "kv",
            }
        results_example.append(entry)

    example = {"results": results_example}
    return (
        "You MUST return JSON in exactly this structure "
        "(one entry per field, same order as schema):\n"
        + json.dumps(example, ensure_ascii=False, indent=2)
    )


def build_messages(
    semantic_info: TableSemanticParserSchema,
    schema: ExtractionSchema,
    table_id: Optional[str] = None,
) -> List[Dict[str, str]]:
    paragraphs_section = _build_paragraphs_section(semantic_info)
    tables_section = _build_tables_section(semantic_info.tables, table_id)
    schema_section = _build_schema_section(schema)
    response_format = _build_response_format(schema)

    user_content = f"""\
## Paragraphs
{paragraphs_section}

## Tables
{tables_section}

## Extraction Schema
{schema_section}

## Response Format
{response_format}
"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
