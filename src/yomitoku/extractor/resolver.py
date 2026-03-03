from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ..schemas import Element, WordPrediction
from ..schemas.table_semantic_parser import (
    CellSchema,
    TableSemanticParserSchema,
)
from ..utils.misc import quad_to_xyxy


@dataclass
class ResolvedElement:
    id: str
    box: List[int]
    contents: str
    label: str = ""


@dataclass
class ResolvedField:
    name: str
    value: Any
    raw_text: str
    elements: List[ResolvedElement] = field(default_factory=list)
    confidence: str = "low"
    source: str = "cell_search"


def build_lookup(
    semantic_info: TableSemanticParserSchema,
) -> Dict[str, Union[CellSchema, Element]]:
    lookup: Dict[str, Union[CellSchema, Element]] = {}

    for table in semantic_info.tables:
        for cell_id, cell in table.cells.items():
            lookup[cell_id] = cell

    for i, paragraph in enumerate(semantic_info.paragraphs):
        pid = paragraph.id if paragraph.id else f"p{i}"
        lookup[pid] = paragraph

    for i, word in enumerate(semantic_info.words):
        lookup[f"w{i}"] = word

    return lookup


def _resolve_element(
    eid: str, lookup: Dict[str, Union[CellSchema, Element, WordPrediction]]
) -> Optional[ResolvedElement]:
    obj = lookup.get(eid)
    if obj is None:
        return None

    if isinstance(obj, WordPrediction):
        box = quad_to_xyxy(obj.points)
        contents = obj.content or ""
    else:
        box = list(obj.box)
        contents = obj.contents or ""
    return ResolvedElement(id=eid, box=box, contents=contents)


def resolve_fields(
    llm_results: List[Dict[str, Any]],
    lookup: Dict[str, Union[CellSchema, Element]],
) -> List[ResolvedField]:
    resolved = []
    for item in llm_results:
        if not isinstance(item, dict):
            continue
        name = item.get("name", "")
        value = item.get("value", "")
        raw_text = item.get("raw_text", "")
        confidence = item.get("confidence", "low")
        source = item.get("source", "cell_search")
        cell_ids = item.get("cell_ids", [])

        elements: List[ResolvedElement] = []

        if isinstance(value, list):
            for row in value:
                if isinstance(row, dict):
                    for col_name, col_val in row.items():
                        if isinstance(col_val, dict):
                            for cid in col_val.get("cell_ids", []):
                                elem = _resolve_element(cid, lookup)
                                if elem:
                                    elem.label = col_name
                                    elements.append(elem)
        elif isinstance(value, dict):
            nested_ids = value.get("cell_ids", [])
            for cid in nested_ids:
                elem = _resolve_element(cid, lookup)
                if elem:
                    elements.append(elem)
            if not cell_ids:
                cell_ids = nested_ids
            value = value.get("value", "")
        else:
            for cid in cell_ids:
                elem = _resolve_element(cid, lookup)
                if elem:
                    elements.append(elem)

        resolved.append(
            ResolvedField(
                name=name,
                value=value,
                raw_text=raw_text,
                elements=elements,
                confidence=confidence,
                source=source,
            )
        )

    return resolved
