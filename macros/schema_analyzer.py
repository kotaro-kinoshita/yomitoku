"""Analyzer utilities for interpreting JSON Schema fragments.

SchemaRenderer calls into this module once per node to normalize schemas
($ref resolution), infer types, and collect displayable facts (constraints,
badges, inline/summary decisions). The helpers avoid side-effects so renderer
logic can reuse the computed analysis safely.
"""

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"
SCALAR_TYPES = {"string", "number", "integer", "boolean", "null"}
ConstraintKind = Literal["literal_list", "literal", "code", "flag"]
ConstraintValue = str | list[str] | None
AdditionalPropsPolicy = Literal["forbidden"]
CONSTRAINT_LABELS: tuple[tuple[str, str], ...] = (
    ("minimum", "Minimum"),
    ("maximum", "Maximum"),
    ("exclusiveMinimum", "Exclusive minimum"),
    ("exclusiveMaximum", "Exclusive maximum"),
    ("multipleOf", "Multipleof"),
    ("minLength", "Minimum length"),
    ("maxLength", "Maximum length"),
    ("minItems", "Minimum items"),
    ("maxItems", "Maximum items"),
    ("minProperties", "Minimum properties"),
    ("maxProperties", "Maximum properties"),
)


@dataclass(frozen=True)
class ConstraintDetail:
    """Structured representation of a constraint to be rendered into HTML later."""

    kind: ConstraintKind
    label: str
    value: ConstraintValue = None


# Immutable bundle returned by analyze(); keeps all analyzer-derived facts together.
@dataclass(frozen=True)
class SchemaNodeAnalysis:
    """Aggregated metadata derived from a schema node."""

    schema: dict[str, Any]  # Dereferenced/normalized schema dict
    type_label: str | None  # Type badge text (inferred/explicit)
    constraints: list[ConstraintDetail]  # Structured constraint facts
    summary_only: bool  # True when header alone is sufficient
    additional_props_policy: AdditionalPropsPolicy | None
    default_label: str | None = None  # Fallback label for array items
    examples: list[str] = field(default_factory=list)


class SchemaAnalyzer:
    """Encapsulates schema normalization, $ref resolution, and metadata extraction."""

    def __init__(self, root_schema: dict[str, Any]) -> None:
        # The root schema object is kept around so any nested $ref pointers can
        # look up definitions by JSON Pointer.
        self.root_schema = root_schema

    def analyze(
        self, schema_def: Any, *, compute_default_label: bool = False
    ) -> SchemaNodeAnalysis:
        """Produce a single analysis bundle for the provided schema node."""
        # Normalize inputs (including resolving $ref pointers and primitive type
        # declarations) into a dict so every downstream step reads a fully
        # materialized schema structure.
        schema = self.normalize_schema(schema_def)
        # Derive the header badge text based on explicit types, unions, or
        # inferred structure so the renderer can label the card succinctly.
        type_label = self.determine_type_label(schema)
        # Aggregate human-readable constraint strings (enums, min/max, formats).
        constraints = self.collect_constraints(schema)
        examples = self.collect_examples(schema)
        # Decide if the node can be represented by its summary alone (no body).
        summary_only = self.is_summary_only(schema)
        # Add an additionalProperties badge when the schema forbids extras.
        additional_props_policy = self.additional_props_policy(
            schema.get("additionalProperties")
        )
        # Provide a fallback label for array items when the caller requests it.
        default_label = (
            self.default_item_label(schema) if compute_default_label else None
        )
        # Package all derived values together so rendering can be a single-step lookup.
        return SchemaNodeAnalysis(
            schema=schema,
            type_label=type_label,
            constraints=constraints,
            summary_only=summary_only,
            additional_props_policy=additional_props_policy,
            default_label=default_label,
            examples=examples,
        )

    def dereference(self, node: dict[str, Any]) -> dict[str, Any]:
        """Resolve any $ref inside the provided node.

        JSON Schema permits nodes to be defined once and referenced many times.
        This helper recursively follows those references until we end up with a
        fully materialized dict.
        """
        if not isinstance(node, dict):
            return node
        if "$ref" not in node:
            return node
        target = self._resolve_pointer(node["$ref"])
        merged: dict[str, Any] = {}
        merged.update(target)
        # Local keys override referenced definitions (JSON Schema merging semantics).
        merged.update({key: value for key, value in node.items() if key != "$ref"})
        return self.dereference(merged)

    def _resolve_pointer(self, pointer: str) -> dict[str, Any]:
        """Resolve a JSON Pointer within the current schema document."""
        if not pointer.startswith("#/"):
            # Only local references are supported because the documentation build
            # bundles schemas into a single file. Anything else would make the output
            # unreliable, so we fail fast with a clear error.
            raise ValueError(f"Only local pointers are supported (got '{pointer}')")

        current: Any = self.root_schema
        for raw_part in pointer[2:].split("/"):
            # JSON Pointer allows ~0 and ~1 escape sequences. Undo them before lookup.
            part = raw_part.replace("~1", "/").replace("~0", "~")
            if isinstance(current, dict):
                current = current[part]
            else:
                # Walking into a non-dict means the pointer itself is invalid.
                raise KeyError(f"Cannot resolve '{part}' in pointer '{pointer}'")

        if isinstance(current, dict):
            return current
        # We only support references to objects because downstream rendering expects dicts.
        raise TypeError(f"Pointer '{pointer}' does not refer to an object")

    def normalize_schema(self, schema_def: Any) -> dict[str, Any]:
        """Ensure every schema reference resolves to a dict for downstream rendering."""
        if isinstance(schema_def, dict):
            return self.dereference(schema_def)
        return {"type": schema_def}

    @staticmethod
    def stringify_literal(value: Any) -> str:
        """Convert literal values to a stable string representation (no HTML)."""
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    def default_item_label(self, schema: dict[str, Any]) -> str:
        """Provide a fallback title for array items."""
        if schema.get("title"):
            return str(schema["title"])
        if self.is_array_schema(schema):
            return "ArrayItem"
        return "Item"

    def collect_constraints(self, schema: dict[str, Any]) -> list[ConstraintDetail]:
        """Collate constraint facts (enum, min/max, etc.).

        The renderer can then decide how to present each fact (code blocks,
        escaped text, etc.) without mixing analysis with markup.
        """
        constraints: list[ConstraintDetail] = []

        if "enum" in schema:
            values = [self.stringify_literal(value) for value in schema["enum"]]
            constraints.append(
                ConstraintDetail("literal_list", "Allowed values", values)
            )
        if "const" in schema:
            constraints.append(
                ConstraintDetail(
                    "literal", "Constant value", self.stringify_literal(schema["const"])
                )
            )
        if "default" in schema:
            constraints.append(
                ConstraintDetail(
                    "literal", "Default", self.stringify_literal(schema["default"])
                )
            )
        if schema.get("format"):
            constraints.append(
                ConstraintDetail("code", "Format", str(schema["format"]))
            )
        if schema.get("pattern"):
            constraints.append(
                ConstraintDetail("code", "Pattern", str(schema["pattern"]))
            )
        for key, label in CONSTRAINT_LABELS:
            if schema.get(key) is not None:
                constraints.append(ConstraintDetail("code", label, str(schema[key])))
        if schema.get("uniqueItems"):
            constraints.append(ConstraintDetail("flag", "Items must be unique"))

        additional_props = schema.get("additionalProperties")
        if additional_props is False:
            constraints.append(
                ConstraintDetail("flag", "Additional properties are not allowed")
            )
        elif isinstance(additional_props, dict):
            constraints.append(
                ConstraintDetail(
                    "flag", "Additional properties must match the nested schema"
                )
            )

        return constraints

    def collect_examples(self, schema: dict[str, Any]) -> list[str]:
        """Collect example values as stringified literals."""
        examples_field = schema.get("examples")
        if not isinstance(examples_field, Iterable) or isinstance(
            examples_field, (str, bytes)
        ):
            return []
        return [self.stringify_literal(example) for example in examples_field]

    def determine_type_label(self, schema: dict[str, Any]) -> str | None:
        """Determine the most informative type label for the header badge."""
        union_label = self._union_type_label(schema)
        if union_label:
            return union_label
        direct = self._normalize_type_label(self._format_type(schema))
        if direct:
            return direct
        inferred = self._normalize_type_label(self._infer_type(schema))
        if inferred:
            return inferred
        return None

    @staticmethod
    def _normalize_type_label(label: str | None) -> str | None:
        """Normalize raw type values and filter out unknown markers."""
        if not label:
            return None
        normalized = str(label).strip()
        if not normalized or normalized.lower() == "unknown":
            return None
        return normalized

    @staticmethod
    def _union_type_label(schema: dict[str, Any]) -> str | None:
        """Return a label representing combination keywords when present."""
        for key in ("anyOf", "oneOf", "allOf"):
            if isinstance(schema.get(key), list):
                return key
        return None

    @staticmethod
    def _type_values(schema: dict[str, Any]) -> list[str]:
        """Return normalized explicit type declarations as a list."""
        type_value = schema.get("type")
        if isinstance(type_value, str):
            return [type_value]
        if isinstance(type_value, Iterable) and not isinstance(
            type_value, (str, bytes)
        ):
            return [str(item) for item in type_value]
        return []

    @staticmethod
    def _format_type(schema: dict[str, Any]) -> str | None:
        """Return the literal `type` field as a displayable string."""
        type_values = SchemaAnalyzer._type_values(schema)
        if not type_values:
            return None
        if len(type_values) == 1:
            return type_values[0]
        return " | ".join(type_values)

    @staticmethod
    def _infer_type(schema: dict[str, Any]) -> str | None:
        """Guess the type from schema structure when no explicit type is set."""
        if "properties" in schema:
            return "object"
        if "items" in schema:
            return "array"
        if "enum" in schema:
            literal_types = {type(item).__name__ for item in schema["enum"]}
            if len(literal_types) == 1:
                return literal_types.pop()
        return None

    def additional_props_policy(
        self, additional_props: Any
    ) -> AdditionalPropsPolicy | None:
        """Return a policy indicator for additionalProperties (for badge rendering)."""
        if additional_props is False:
            return "forbidden"
        return None

    def is_summary_only(self, schema: dict[str, Any]) -> bool:
        """Return True when the schema can be represented by the header alone."""
        return self.is_scalar(schema) or self._is_combination_of_scalars(schema)

    def _is_combination_of_scalars(self, schema: dict[str, Any]) -> bool:
        """Check if a combination keyword contains only scalar alternatives.

        Combination keywords (allOf/anyOf/oneOf/not) can describe complex
        structures. When every branch resolves to a scalar, we can keep the UI
        compact by skipping nested cards. This helper performs that check so the
        renderer can make a quick yes/no decision.
        """
        combo_keys = ("allOf", "anyOf", "oneOf")
        for key in combo_keys:
            clauses = schema.get(key)
            if isinstance(clauses, list) and clauses:
                all_scalar = True
                for clause in clauses:
                    normalized = (
                        clause if isinstance(clause, dict) else {"type": clause}
                    )
                    if not self.is_summary_only(self.normalize_schema(normalized)):
                        all_scalar = False
                        break
                if all_scalar:
                    return True
        not_clause = schema.get("not")
        if isinstance(not_clause, dict):
            return self.is_summary_only(self.normalize_schema(not_clause))
        return False

    def is_scalar(self, schema: dict[str, Any]) -> bool:
        """Return True for schemas that ultimately represent scalar values."""
        type_values = self._type_values(schema)
        if type_values:
            return all(value in SCALAR_TYPES for value in type_values)
        if schema.get("properties") or schema.get("items"):
            return False
        inferred = self._infer_type(schema)
        return inferred in SCALAR_TYPES if inferred else False

    def is_array_schema(self, schema: dict[str, Any]) -> bool:
        """Return True if the schema definitely describes an array."""
        type_values = self._type_values(schema)
        if type_values and "array" in type_values:
            return True
        if "items" in schema:
            return True
        inferred = self._infer_type(schema)
        return inferred == "array"
