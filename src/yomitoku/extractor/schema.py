from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel, Field


class ColumnSchema(BaseModel):
    name: str = Field(..., description="Column name (used as output key)")
    description: str = Field(
        "", description="Human-readable column description for matching"
    )
    cell_id: Optional[str] = Field(
        None, description="Cell ID for direct header cell matching"
    )
    bbox: Optional[List[int]] = Field(
        None, description="Bounding box [x1, y1, x2, y2] for header cell matching"
    )
    type: Literal[
        "string", "number", "date", "alphanumeric", "hiragana", "katakana"
    ] = Field("string", description="Value type")
    normalize: Optional[str] = Field(None, description="Normalization rule name")


class FieldSchema(BaseModel):
    name: str = Field(..., description="Field name (used as output key)")
    description: str = Field("", description="Human-readable field description")
    cell_id: Optional[str] = Field(None, description="Cell ID for direct cell lookup")
    bbox: Optional[List[int]] = Field(
        None, description="Bounding box [x1, y1, x2, y2] for cell search"
    )
    regex: Optional[str] = Field(None, description="Regex pattern for value extraction")
    type: Literal[
        "string", "number", "date", "alphanumeric", "hiragana", "katakana"
    ] = Field("string", description="Value type (for scalar fields)")
    structure: Literal["scalar", "kv", "table"] = Field(
        "scalar", description="Data structure: scalar (or kv) or table"
    )
    normalize: Optional[str] = Field(
        None, description="Normalization rule name (for scalar fields)"
    )
    columns: Optional[List[ColumnSchema]] = Field(
        None, description="Column definitions (for table fields)"
    )


class ExtractionSchema(BaseModel):
    fields: List[FieldSchema] = Field(..., description="List of fields to extract")

    @classmethod
    def from_yaml(cls, path: str) -> "ExtractionSchema":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
