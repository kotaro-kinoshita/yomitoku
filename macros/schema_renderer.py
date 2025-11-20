"""Traverse JSON Schemas and render nested cards."""

import html
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from macros.schema_analyzer import SCHEMA_DIR, SchemaAnalyzer
from macros.schema_html_builder import SchemaHTMLBuilder

@dataclass
class NodeRenderMeta:
    """Aggregated metadata for rendering a schema node."""

    schema: dict[str, Any]
    title: str
    type_label: str | None
    description: str
    summary_desc: str
    badges: list[str]
    anchor: str
    allow_inline_title: bool


class SchemaRenderer:
    """Coordinates schema loading, traversal, and HTML generation."""

    def __init__(self, schema_dir: Path = SCHEMA_DIR) -> None:
        """Initialize renderer state and shared helpers."""
        self.schema_dir = schema_dir
        self._cache: dict[str, dict[str, Any]] = {}
        self.root_title: str | None = None
        self._analyzer: SchemaAnalyzer | None = None
        self._html_builder: SchemaHTMLBuilder | None = None

    def render(self, schema_name: str) -> str:
        """Render a schema by name into nested card markup."""
        schema = self._load(schema_name)
        title = schema.get("title") or schema_name

        assert isinstance(title, str)

        self._initialize_components(schema, title)
        parts = [
            # Root wrapper ensures CSS can style the entire schema tree at once.
            '<div class="schema-card-tree">',
            self._render_node(
                schema,
                display_name=title,
                segments=["root"],
                required=False,
                root_schema=schema,
            ),
            "</div>",
        ]
        return "\n".join(parts)

    def _initialize_components(self, schema: dict[str, Any], title: str) -> None:
        """Set up analyzer and HTML builder for the active schema."""
        self.root_title = title
        self._analyzer = SchemaAnalyzer(schema)
        self._html_builder = SchemaHTMLBuilder(self.root_title, self._analyzer)

    def _render_node(
        self,
        node: dict[str, Any],
        *,
        display_name: str | None,
        segments: Sequence[str],
        required: bool,
        root_schema: dict[str, Any],
        inline: bool = False,
        open_card: bool = False,
        force_summary_only: bool = False,
    ) -> str:
        """Render a single schema object (or reference) into a card."""
        # Resolve $ref chains before doing any expensive rendering work.
        schema = self._resolve_schema_node(node)
        meta = self._build_node_meta(schema, display_name, segments, required)
        body_html = self._build_node_body(
            meta,
            segments,
            root_schema,
            inline,
            force_summary_only,
        )
        summary_inner, summary_badge_only = self._build_node_summary(meta, inline)
        permalink_html = self._require_html_builder().permalink_link(meta.anchor, meta.title)
        return self._render_card_output(
            inline=inline,
            anchor=meta.anchor,
            summary_inner=summary_inner,
            body_html=body_html,
            summary_badge_only=summary_badge_only,
            open_card=open_card,
            permalink_html=permalink_html,
        )

    def _render_children(
        self, schema: dict[str, Any], segments: Sequence[str], root_schema: dict[str, Any]
    ) -> str:
        """Render child cards for properties, items, and combinators."""
        base_segments = list(segments)
        child_sections: list[str] = []
        # Maintain this order so properties show up before array items/combinators.
        child_sections.extend(self._render_properties(schema, base_segments, root_schema))
        child_sections.extend(self._render_items(schema, base_segments, root_schema))
        child_sections.extend(
            self._render_additional_properties(schema, base_segments, root_schema)
        )
        child_sections.extend(self._render_combinations(schema, base_segments, root_schema))
        if not child_sections:
            return ""
        return self._require_html_builder().render_children_container(child_sections)

    def _render_properties(
        self, schema: dict[str, Any], segments: list[str], root_schema: dict[str, Any]
    ) -> list[str]:
        """Render the child cards for every property on an object schema."""
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return []
        # JSON Schema lists required fields by name, so convert to a set for quick lookups.
        required_set = set(schema.get("required") or [])
        children: list[str] = []
        for prop_name, prop_schema in properties.items():
            child_segments = list(segments) + [str(prop_name)]
            children.append(
                self._render_node(
                    prop_schema,
                    display_name=str(prop_name),
                    segments=child_segments,
                    required=str(prop_name) in required_set,
                    root_schema=root_schema,
                )
            )
        return children

    def _render_items(
        self, schema: dict[str, Any], segments: list[str], root_schema: dict[str, Any]
    ) -> list[str]:
        """Render the schema for array items (single schema or tuple definitions)."""
        analyzer = self._require_analyzer()
        items_schema = schema.get("items")
        if not items_schema:
            return []
        if isinstance(items_schema, list):
            # Tuple validation: every position has its own schema.
            return self._render_tuple_items(items_schema, segments, root_schema)
        normalized = analyzer.normalize_schema(items_schema)
        label = analyzer.default_item_label(normalized)
        child_segments = list(segments) + ["[item]"]
        if not analyzer.is_summary_only(normalized):
            child_segments.append(label)
        return [
            self._render_item_node(
                normalized,
                segments=child_segments,
                root_schema=root_schema,
                label=label,
            )
        ]

    def _render_tuple_items(
        self,
        tuple_items: Sequence[Any],
        segments: list[str],
        root_schema: dict[str, Any],
    ) -> list[str]:
        """Render tuple-style arrays where each position has a dedicated schema."""
        analyzer = self._require_analyzer()
        rendered: list[str] = []
        for index, item_schema in enumerate(tuple_items, start=1):
            normalized = analyzer.normalize_schema(item_schema)
            label = f"{analyzer.default_item_label(normalized)} {index}"
            child_segments = list(segments) + [f"[item{index}]"]
            if not analyzer.is_summary_only(normalized):
                child_segments.append(label)
            rendered.append(
                self._render_item_node(
                    normalized,
                    segments=child_segments,
                    root_schema=root_schema,
                    label=label,
                )
            )
        return rendered

    def _render_item_node(
        self,
        schema: dict[str, Any],
        *,
        segments: Sequence[str],
        root_schema: dict[str, Any],
        label: str,
    ) -> str:
        """Render a single array item schema."""
        analyzer = self._require_analyzer()
        inline = analyzer.is_summary_only(schema)
        open_card = False
        if label == "ArrayItem":
            # Force non-inline rendering so anonymous array definitions stay discoverable.
            inline = False
            open_card = True
        return self._render_node(
            schema,
            display_name=label,
            segments=list(segments),
            required=False,
            root_schema=root_schema,
            inline=inline,
            open_card=open_card,
        )

    def _render_additional_properties(
        self, schema: dict[str, Any], segments: list[str], root_schema: dict[str, Any]
    ) -> list[str]:
        """Render schemas for additionalProperties if present."""
        additional_props = schema.get("additionalProperties")
        if isinstance(additional_props, dict):
            child_segments = list(segments) + ["[*]"]
            return [
                self._render_node(
                    additional_props,
                    display_name="Additional property",
                    segments=child_segments,
                    required=False,
                    root_schema=root_schema,
                )
            ]
        return []

    def _render_combinations(
        self, schema: dict[str, Any], segments: list[str], root_schema: dict[str, Any]
    ) -> list[str]:
        """Render composition keywords like allOf/anyOf/oneOf/not."""
        combinations: list[str] = []
        combo_defs = [
            ("allOf", "All of"),
            ("anyOf", "Any of"),
            ("oneOf", "One of"),
        ]
        for key, label in combo_defs:
            clauses = schema.get(key)
            if isinstance(clauses, list):
                for index, clause in enumerate(clauses, 1):
                    combinations.append(
                        self._render_node(
                            clause,
                            display_name=f"{label} {index}",
                            segments=list(segments) + [f"{key}[{index}]"],
                            required=False,
                            root_schema=root_schema,
                            inline=True,
                            # Force inline layout so we do not get nested accordion toggles.
                            force_summary_only=True,
                        )
                    )

        not_clause = schema.get("not")
        if isinstance(not_clause, dict):
            combinations.append(
                self._render_node(
                    not_clause,
                    display_name="Not",
                    segments=list(segments) + ["not"],
                    required=False,
                    root_schema=root_schema,
                    inline=True,
                    # NOT clauses rarely have bodies; force summary-only to stay compact.
                    force_summary_only=True,
                )
            )
        return combinations

    def _resolve_schema_node(self, node: dict[str, Any]) -> dict[str, Any]:
        """Dereference nodes and ensure we have a schema dict."""
        return self._require_analyzer().dereference(node)

    def _build_node_meta(
        self,
        schema: dict[str, Any],
        display_name: str | None,
        segments: Sequence[str],
        required: bool,
    ) -> NodeRenderMeta:
        """Collect metadata needed across summary/body rendering."""
        analyzer = self._require_analyzer()
        html_builder = self._require_html_builder()
        type_label = analyzer.determine_type_label(schema)
        title = self._resolve_title(schema, display_name, segments)
        anchor = html_builder.anchor(segments)
        description = schema.get("description", "").strip()
        badges = html_builder.build_badges(schema, type_label, required)
        summary_desc = self._format_summary_description(description)
        allow_inline_title = bool(schema.get("title") or display_name)
        return NodeRenderMeta(
            schema=schema,
            title=title,
            type_label=type_label,
            description=description,
            summary_desc=summary_desc,
            badges=badges,
            anchor=anchor,
            allow_inline_title=allow_inline_title,
        )

    def _build_node_body(
        self,
        meta: NodeRenderMeta,
        segments: Sequence[str],
        root_schema: dict[str, Any],
        inline: bool,
        force_summary_only: bool,
    ) -> str:
        """Build the full body HTML for the current node."""
        analyzer = self._require_analyzer()
        html_builder = self._require_html_builder()
        summary_only_inline = inline and (
            force_summary_only or analyzer.is_summary_only(meta.schema)
        )
        # Constraints and description metadata feed into the body sections.
        constraints = analyzer.collect_constraints(meta.schema)
        body_sections = html_builder.build_body_sections(
            segments,
            meta.type_label,
            meta.description,
            constraints,
            meta.schema,
        )
        children_html = self._render_children(meta.schema, segments, root_schema)
        return html_builder.build_body_html(
            body_sections,
            children_html,
            summary_only_inline,
        )

    def _build_node_summary(self, meta: NodeRenderMeta, inline: bool) -> tuple[str, bool]:
        """Build the summary HTML and track whether it only contains badges."""
        html_builder = self._require_html_builder()
        return html_builder.build_summary(
            title=meta.title,
            summary_desc=meta.summary_desc,
            badges=meta.badges,
            inline=inline,
            allow_inline_title=meta.allow_inline_title,
        )

    @staticmethod
    def _format_summary_description(description: str) -> str:
        """Render primary line normally and secondary lines using the small style."""
        if not description:
            return ""
        lines = [line.strip() for line in description.splitlines() if line.strip()]
        if not lines:
            return ""
        formatted_lines = [
            f'<span class="schema-card-summary-desc">{html.escape(line)}</span>'
            for line in lines
        ]
        return "<br />".join(formatted_lines)

    def _render_card_output(
        self,
        *,
        inline: bool,
        anchor: str,
        summary_inner: str,
        body_html: str,
        summary_badge_only: bool,
        open_card: bool,
        permalink_html: str,
    ) -> str:
        """Dispatch to inline or block card renderers."""
        html_builder = self._require_html_builder()
        if inline:
            # Inline cards (used for scalar array items) render without collapsible details
            # to reduce visual noise inside lists.
            return html_builder.render_inline_card(anchor, summary_inner, body_html, permalink_html)
        return html_builder.render_block_card(
            anchor,
            summary_inner,
            body_html,
            summary_badge_only,
            open_card,
            permalink_html,
        )

    def _resolve_title(
        self, schema: dict[str, Any], display_name: str | None, segments: Sequence[str]
    ) -> str:
        """Use explicit names when provided, otherwise fall back to schema titles."""
        if display_name is not None:
            return display_name
        if schema.get("title"):
            return str(schema["title"])
        return segments[-1]

    def _load(self, schema_name: str) -> dict[str, Any]:
        """Load a schema JSON file and cache it for subsequent renders."""
        filename = (
            f"{schema_name}.json" if not schema_name.endswith(".json") else schema_name
        )
        if filename not in self._cache:
            schema_path = self.schema_dir / filename
            if not schema_path.exists():
                raise FileNotFoundError(f"Schema '{filename}' not found in {self.schema_dir}")
            # Cache parsed JSON so repeated macro calls (or nested refs) avoid disk I/O.
            self._cache[filename] = json.loads(schema_path.read_text(encoding="utf-8"))
        return self._cache[filename]

    def _require_analyzer(self) -> SchemaAnalyzer:
        """Return the initialized analyzer, raising if it is unavailable."""
        if self._analyzer is None:
            raise RuntimeError("SchemaAnalyzer has not been initialized")
        return self._analyzer

    def _require_html_builder(self) -> SchemaHTMLBuilder:
        """Return the initialized HTML builder, raising if it is unavailable."""
        if self._html_builder is None:
            raise RuntimeError("SchemaHTMLBuilder has not been initialized")
        return self._html_builder
