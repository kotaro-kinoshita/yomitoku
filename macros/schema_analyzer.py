"""Utility helpers for interpreting JSON Schema fragments during rendering.

SchemaRenderer leans on this module any time it needs to interpret or normalize
JSON Schema content (e.g. following $ref pointers, extracting human readable
badges, or inferring data types). Every helper aims to keep side-effects
minimal so the renderer can call them freely without worrying about state.
"""

import html
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"
SCALAR_TYPES = {"string", "number", "integer", "boolean", "null"}
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


def format_literal(value: Any) -> str:
    """Render literal values (dict/list/primitives) as inline code.

    Many call sites (enum, const, examples, etc.) present raw Python objects to
    the user. Wrapping them in <code> ensures consistent styling while escaping
    HTML so we never output unsafe content.
    """
    if isinstance(value, (dict, list)):
        return f"<code>{html.escape(json.dumps(value, ensure_ascii=False))}</code>"
    return f"<code>{html.escape(str(value))}</code>"


class SchemaAnalyzer:
    """Encapsulates schema normalization, $ref resolution, and metadata extraction."""

    def __init__(self, root_schema: dict[str, Any]) -> None:
        # The root schema object is kept around so any nested $ref pointers can
        # look up definitions by JSON Pointer.
        self.root_schema = root_schema

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

    def default_item_label(self, schema: dict[str, Any]) -> str:
        """Provide a fallback title for array items."""
        if schema.get("title"):
            return str(schema["title"])
        if self.is_array_schema(schema):
            return "ArrayItem"
        return "Item"

    def collect_constraints(self, schema: dict[str, Any]) -> list[str]:
        """Collate constraint strings (enum, min/max, etc.) for display.

        The MkDocs UI shows human readable bullet lists explaining everything
        from allowed values to length limits.
        """
        constraints: list[str] = []

        if "enum" in schema:
            values = ", ".join(format_literal(value) for value in schema["enum"])
            constraints.append(f"Allowed values: {values}")
        if "const" in schema:
            constraints.append(f"Constant value: {format_literal(schema['const'])}")
        if "default" in schema:
            constraints.append(f"Default: {format_literal(schema['default'])}")
        if schema.get("format"):
            constraints.append(f"Format: <code>{html.escape(str(schema['format']))}</code>")
        if schema.get("pattern"):
            constraints.append(f"Pattern: <code>{html.escape(str(schema['pattern']))}</code>")
        for key, label in CONSTRAINT_LABELS:
            if schema.get(key) is not None:
                constraints.append(
                    f"{label}: <code>{html.escape(str(schema[key]))}</code>"
                )
        if schema.get("uniqueItems"):
            constraints.append("Items must be unique")

        additional_props = schema.get("additionalProperties")
        if additional_props is False:
            constraints.append("Additional properties are not allowed")
        elif isinstance(additional_props, dict):
            constraints.append("Additional properties must match the nested schema")

        return constraints

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
        if isinstance(type_value, Iterable) and not isinstance(type_value, (str, bytes)):
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

    def additional_badge(self, additional_props: Any) -> str | None:
        """Return badge HTML describing additional property behavior."""
        if additional_props is False:
            text = "No Additional Props"
        else:
            return None
        # Badge text mirrors the MkDocs cards, so keep wording consistent here.
        return f'<span class="schema-badge schema-badge--additional">{html.escape(text)}</span>'

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
                    normalized = clause if isinstance(clause, dict) else {"type": clause}
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
