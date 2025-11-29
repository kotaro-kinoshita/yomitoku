"""Traverse JSON Schemas and render nested cards."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from macros.schema_analyzer import SCHEMA_DIR, SchemaAnalyzer, SchemaNodeAnalysis
from macros.schema_html_builder import SchemaHTMLBuilder


# UI-facing metadata derived from analyzer output plus render context.
@dataclass
class NodeRenderMeta:
    """Aggregated metadata for rendering a schema node."""

    schema: dict[str, Any]  # Fully resolved schema dict for this node
    title: str  # Display title resolved from schema/display_name/path
    type_label: str | None  # Human-friendly type label badge
    description: str  # Raw description text from schema
    summary_desc: str  # Formatted summary description HTML (escaped)
    badges: list[str]  # Render-ready badge HTML snippets
    anchor: str  # Stable anchor id for permalinks/breadcrumbs
    allow_inline_title: bool  # Whether title may render in inline summaries


@dataclass
class RenderNodeParams:
    """Inputs required to render a node."""

    node: dict[str, Any]  # Target schema node
    display_name: str | None  # Preferred display name for the node
    segments: Sequence[str]  # Anchor/breadcrumb path segments
    required: bool  # Whether the node is required by its parent
    inline: bool = False  # Render as inline card
    open_card: bool = False  # Start expanded by default
    force_summary_only: bool = False  # Force summary-only rendering
    analysis: SchemaNodeAnalysis | None = None  # Optional precomputed analysis


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
                RenderNodeParams(
                    node=schema,
                    display_name=title,
                    segments=["root"],
                    required=False,
                )
            ),
            "</div>",
        ]
        return "\n".join(parts)

    def _initialize_components(self, schema: dict[str, Any], title: str) -> None:
        """Set up analyzer and HTML builder for the active schema."""
        self.root_title = title
        self._analyzer = SchemaAnalyzer(schema)
        self._html_builder = SchemaHTMLBuilder(self.root_title)

    def _render_node(
        self,
        params: RenderNodeParams,
    ) -> str:
        """Render a single schema object (or reference) into a card."""
        analyzer = self._require_analyzer()
        # Perform analysis once per node, but reuse when provided.
        analysis = params.analysis or analyzer.analyze(params.node)

        # Derive render-specific metadata (titles/anchors/badges) from analysis.
        meta = self._build_node_meta(
            analysis, params.display_name, params.segments, params.required
        )
        # Decide collapse/inline behavior up front so it is clear what drives layout.
        summary_only_inline = self._summary_only_inline(
            analysis, params.inline, params.force_summary_only
        )

        # Render children explicitly here so the recursive tree flow stays visible.
        children_html = self._render_children(meta.schema, params.segments)
        # Build body/summary HTML and assemble the card.
        body_html = self._build_node_body(
            meta,
            analysis,
            params.segments,
            children_html,
            summary_only_inline,
        )
        # Build the summary (title/description/badges) and permalink together so
        # the final render step knows exactly what header content to wrap.
        summary_inner, summary_badge_only = self._build_node_summary(
            meta, params.inline
        )
        # Generate an accessible permalink that lives outside <summary> so the
        # accordion toggle remains usable while still providing deep links.
        permalink_html = self._require_html_builder().permalink_link(
            meta.anchor, meta.title
        )
        # Delegate to the appropriate card renderer (inline vs collapsible block)
        # now that all pieces—summary, body, anchor, flags—have been computed.
        return self._render_card_output(
            inline=params.inline,
            anchor=meta.anchor,
            summary_inner=summary_inner,
            body_html=body_html,
            summary_badge_only=summary_badge_only,
            open_card=params.open_card,
            permalink_html=permalink_html,
        )

    def _render_children(
        self,
        schema: dict[str, Any],
        segments: Sequence[str],
    ) -> str:
        """Render child cards for properties, items, and combinators."""
        requests: list[RenderNodeParams] = []
        base_segments = list(segments)
        # Maintain this order so properties show up before array items/combinators.
        requests.extend(self._property_child_requests(schema, base_segments))
        requests.extend(self._item_child_requests(schema, base_segments))
        requests.extend(self._additional_property_requests(schema, base_segments))
        requests.extend(self._combination_child_requests(schema, base_segments))

        if not requests:
            return ""
        rendered = [self._render_node(params) for params in requests]
        return self._require_html_builder().render_children_container(rendered)

    def _property_child_requests(
        self, schema: dict[str, Any], segments: list[str]
    ) -> list[RenderNodeParams]:
        """Describe child cards for every property on an object schema."""
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return []
        # JSON Schema lists required fields by name, so convert to a set for quick lookups.
        required_set = set(schema.get("required") or [])
        children: list[RenderNodeParams] = []
        for prop_name, prop_schema in properties.items():
            child_segments = list(segments) + [str(prop_name)]
            children.append(
                RenderNodeParams(
                    node=prop_schema,
                    display_name=str(prop_name),
                    segments=child_segments,
                    required=str(prop_name) in required_set,
                )
            )
        return children

    def _item_child_requests(
        self, schema: dict[str, Any], segments: list[str]
    ) -> list[RenderNodeParams]:
        """Describe child cards for array items (single schema or tuple definitions)."""
        items_schema = schema.get("items")
        if not items_schema:
            return []
        if isinstance(items_schema, list):
            # Tuple validation: every position has its own schema.
            return self._tuple_item_child_requests(items_schema, segments)
        # For simple items, analyze once and reuse the computed label/flags.
        analysis = self._require_analyzer().analyze(
            items_schema, compute_default_label=True
        )
        label = analysis.default_label or "Item"
        child_segments = list(segments) + ["[item]"]
        child_segments_with_label = child_segments + [label]
        inline = analysis.summary_only
        open_card = label == "ArrayItem"
        if open_card:
            inline = False
        return [
            RenderNodeParams(
                node=analysis.schema,
                display_name=label,
                segments=child_segments_with_label,
                required=False,
                inline=inline,
                open_card=open_card,
                force_summary_only=False,
                analysis=analysis,
            )
        ]

    def _tuple_item_child_requests(
        self, tuple_items: Sequence[Any], segments: list[str]
    ) -> list[RenderNodeParams]:
        """Describe tuple-style array items where each position has a dedicated schema."""
        requests: list[RenderNodeParams] = []
        for index, item_schema in enumerate(tuple_items, start=1):
            # Each tuple position gets its own analysis to capture titles/summary flags.
            analysis = self._require_analyzer().analyze(
                item_schema, compute_default_label=True
            )
            default_label = analysis.default_label or "Item"
            label = f"{default_label} {index}"
            child_segments = list(segments) + [f"[item{index}]"]
            child_segments_with_label = child_segments + [label]
            inline = analysis.summary_only
            requests.append(
                RenderNodeParams(
                    node=analysis.schema,
                    display_name=label,
                    segments=child_segments_with_label,
                    required=False,
                    inline=inline,
                    open_card=False,
                    force_summary_only=False,
                    analysis=analysis,
                )
            )
        return requests

    def _additional_property_requests(
        self, schema: dict[str, Any], segments: list[str]
    ) -> list[RenderNodeParams]:
        """Describe schemas for additionalProperties if present."""
        additional_props = schema.get("additionalProperties")
        if isinstance(additional_props, dict):
            child_segments = list(segments) + ["[*]"]
            return [
                RenderNodeParams(
                    node=additional_props,
                    display_name="Additional property",
                    segments=child_segments,
                    required=False,
                )
            ]
        return []

    def _combination_child_requests(
        self, schema: dict[str, Any], segments: list[str]
    ) -> list[RenderNodeParams]:
        """Describe composition keyword children like allOf/anyOf/oneOf/not."""
        combinations: list[RenderNodeParams] = []
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
                        RenderNodeParams(
                            node=clause,
                            display_name=f"{label} {index}",
                            segments=list(segments) + [f"{key}[{index}]"],
                            required=False,
                            inline=True,
                            # Force inline layout so we do not get nested accordion toggles.
                            force_summary_only=True,
                        )
                    )

        not_clause = schema.get("not")
        if isinstance(not_clause, dict):
            combinations.append(
                RenderNodeParams(
                    node=not_clause,
                    display_name="Not",
                    segments=list(segments) + ["not"],
                    required=False,
                    inline=True,
                    # NOT clauses rarely have bodies; force summary-only to stay compact.
                    force_summary_only=True,
                )
            )
        return combinations

    @staticmethod
    def _summary_only_inline(
        analysis: SchemaNodeAnalysis, inline: bool, force_summary_only: bool
    ) -> bool:
        """Compute whether a node should render as summary-only inline content."""
        return inline and (force_summary_only or analysis.summary_only)

    def _build_node_meta(
        self,
        analysis: SchemaNodeAnalysis,
        display_name: str | None,
        segments: Sequence[str],
        required: bool,
    ) -> NodeRenderMeta:
        """Collect metadata needed across summary/body rendering."""
        html_builder = self._require_html_builder()
        schema = analysis.schema
        title = self._resolve_title(schema, display_name, segments)
        anchor = html_builder.anchor(segments)
        description = schema.get("description", "").strip()
        badges = html_builder.build_badges(
            analysis.type_label, required, analysis.additional_badge
        )
        summary_desc = html_builder.format_summary_description(description)
        allow_inline_title = bool(schema.get("title") or display_name)
        return NodeRenderMeta(
            schema=schema,
            title=title,
            type_label=analysis.type_label,
            description=description,
            summary_desc=summary_desc,
            badges=badges,
            anchor=anchor,
            allow_inline_title=allow_inline_title,
        )

    def _build_node_body(
        self,
        meta: NodeRenderMeta,
        analysis: SchemaNodeAnalysis,
        segments: Sequence[str],
        children_html: str,
        summary_only_inline: bool,
    ) -> str:
        """Build the full body HTML for the current node."""
        html_builder = self._require_html_builder()
        # Constraints and description metadata feed into the body sections.
        body_sections = html_builder.build_body_sections(
            segments,
            meta.type_label,
            meta.description,
            analysis.constraints,
            meta.schema,
        )
        return html_builder.build_body_html(
            body_sections,
            children_html,
            summary_only_inline,
        )

    def _build_node_summary(
        self, meta: NodeRenderMeta, inline: bool
    ) -> tuple[str, bool]:
        """Build the summary HTML and track whether it only contains badges."""
        html_builder = self._require_html_builder()
        return html_builder.build_summary(
            title=meta.title,
            summary_desc=meta.summary_desc,
            badges=meta.badges,
            inline=inline,
            allow_inline_title=meta.allow_inline_title,
        )

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
            return html_builder.render_inline_card(
                anchor, summary_inner, body_html, permalink_html
            )
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
                raise FileNotFoundError(
                    f"Schema '{filename}' not found in {self.schema_dir}"
                )
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
