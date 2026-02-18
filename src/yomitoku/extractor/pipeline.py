import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from ..schemas.table_semantic_parser import TableSemanticParserSchema
from ..utils.logger import set_logger
from .llm_client import call_llm
from .normalizer import apply_normalize
from .prompt import build_messages
from .resolver import ResolvedField, build_lookup, resolve_fields
from .schema import ExtractionSchema
from .visualizer import extraction_visualizer

logger = set_logger(__name__, "INFO")


def _normalize_resolved_fields(
    fields: List[ResolvedField],
    schema: ExtractionSchema,
    skip_normalize: bool = False,
) -> List[ResolvedField]:
    if skip_normalize:
        return fields

    schema_map = {f.name: f for f in schema.fields}

    for rf in fields:
        fs = schema_map.get(rf.name)
        if fs is None:
            continue

        if fs.structure in ("scalar", "kv"):
            if isinstance(rf.value, str):
                rf.value = apply_normalize(rf.value, fs.normalize)
        elif fs.structure == "table" and isinstance(rf.value, list):
            col_map = {}
            if fs.columns:
                col_map = {c.name: c for c in fs.columns}
            for row in rf.value:
                if not isinstance(row, dict):
                    continue
                for col_name, col_val in row.items():
                    col_schema = col_map.get(col_name)
                    if (
                        col_schema
                        and col_schema.normalize
                        and isinstance(col_val, dict)
                    ):
                        col_val["value"] = apply_normalize(
                            col_val.get("value", ""), col_schema.normalize
                        )

    return fields


def _build_output(fields: List[ResolvedField]) -> Dict[str, Any]:
    output: Dict[str, Any] = {"fields": {}, "metadata": {"schema_version": "1.0"}}

    for rf in fields:
        box_lookup = {elem.id: elem.box for elem in rf.elements}
        contents_lookup = {elem.id: elem.contents for elem in rf.elements}

        if isinstance(rf.value, list):
            records = []
            for row in rf.value:
                if not isinstance(row, dict):
                    records.append(row)
                    continue
                enriched_row = {}
                for col_name, col_val in row.items():
                    if isinstance(col_val, dict):
                        cell_ids = col_val.get("cell_ids", [])
                        bboxes = [
                            box_lookup[cid] for cid in cell_ids if cid in box_lookup
                        ]
                        raw_text = " ".join(
                            contents_lookup[cid]
                            for cid in cell_ids
                            if cid in contents_lookup
                        )
                        enriched_row[col_name] = {
                            **col_val,
                            "raw_text": raw_text,
                            "bboxes": bboxes,
                        }
                    else:
                        enriched_row[col_name] = col_val
                records.append(enriched_row)

            output["fields"][rf.name] = {
                "structure": "table",
                "records": records,
                "source": rf.source,
            }
        else:
            bboxes = [elem.box for elem in rf.elements]
            cell_ids = [elem.id for elem in rf.elements]

            output["fields"][rf.name] = {
                "structure": "kv",
                "value": rf.value,
                "raw_text": rf.raw_text,
                "confidence": rf.confidence,
                "source": rf.source,
                "cell_ids": cell_ids,
                "bboxes": bboxes,
            }

    return output


def _build_simple_output(fields: List[ResolvedField]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}

    for rf in fields:
        if isinstance(rf.value, list):
            records = []
            for row in rf.value:
                if not isinstance(row, dict):
                    records.append(row)
                    continue
                simple_row = {}
                for col_name, col_val in row.items():
                    if isinstance(col_val, dict):
                        simple_row[col_name] = col_val.get("value", "")
                    else:
                        simple_row[col_name] = col_val
                records.append(simple_row)
            output[rf.name] = records
        elif isinstance(rf.value, dict):
            output[rf.name] = rf.value.get("value", "")
        else:
            output[rf.name] = rf.value

    return output


def run_extraction(
    semantic_info: TableSemanticParserSchema,
    img: np.ndarray,
    schema: ExtractionSchema,
    model: str,
    api_base: str = "http://localhost:8000/v1",
    api_key: str = "",
    temperature: float = 0.0,
    max_tokens: int = 4096,
    table_id: Optional[str] = None,
    no_normalize: bool = False,
    prompt_only: bool = False,
    visualize: bool = False,
    simple: bool = False,
    outdir: str = "results",
    filename: str = "output",
) -> Dict[str, Any]:
    messages = build_messages(semantic_info, schema, table_id=table_id)

    if prompt_only:
        for msg in messages:
            logger.info(f"[{msg['role']}]\n{msg['content']}")
        return {"prompt": messages}

    logger.info("Calling LLM API...")
    llm_response = call_llm(
        messages=messages,
        model=model,
        api_base=api_base,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    results = llm_response.get("results", [])
    if not results:
        logger.warning("LLM returned empty results")

    lookup = build_lookup(semantic_info)
    resolved = resolve_fields(results, lookup)
    resolved = _normalize_resolved_fields(resolved, schema, skip_normalize=no_normalize)

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
