"""HTML fragment builder used by SchemaRenderer."""

import html
import re
from collections.abc import Sequence
from typing import Any

from macros.schema_analyzer import SchemaAnalyzer, format_literal


class SchemaHTMLBuilder:
    """Generates HTML fragments (summary, badges, sections) for schema cards.

    The renderer constructs raw metadata for every node, then hands it off to
    this builder. Keeping markup creation in a dedicated class makes it easier
    to tweak HTML/CSS without touching traversal logic.
    """

    def __init__(self, root_title: str, analyzer: SchemaAnalyzer) -> None:
        self.root_title = root_title
        self.analyzer = analyzer
        # Pre-compute the parent slug since every anchor path begins with it.
        self._root_slug = self._slug(self.root_title or "root")

    def anchor(self, segments: Sequence[str]) -> str:
        """Build a stable anchor id from the current segment path.

        Anchors are used for permalinks and breadcrumbs, so they must be stable
        even if the schema order changes. We slugify each path segment and join
        them with dashes, skipping placeholder markers like "[item]".
        """
        slug_parts: list[str] = []
        for idx, segment in enumerate(segments):
            if idx == 0:
                cleaned = self._root_slug
            elif segment == "[item]":
                # Do not include the placeholder segment so anchors stay compact.
                continue
            else:
                cleaned = self._slug(segment)
            if cleaned:
                slug_parts.append(cleaned)
        return "schema-" + "-".join(slug_parts or ["section"])

    def build_badges(
        self, schema: dict[str, Any], type_label: str | None, required: bool
    ) -> list[str]:
        """Compose the badge list that sits in the card header.

        Cards may show multiple badges (required/type/additional properties).
        Returning pure HTML snippets keeps `build_summary` focused on layout.
        """
        badges: list[str] = []
        if required:
            badges.append('<span class="schema-badge schema-badge--required">Required</span>')
        additional_badge = self.analyzer.additional_badge(schema.get("additionalProperties"))
        if additional_badge:
            badges.append(additional_badge)
        if type_label:
            badges.append(
                f'<span class="schema-badge schema-badge--type">{html.escape(type_label)}</span>'
            )
        return [self._wrap_badge_container(badge) for badge in badges]

    def build_body_sections(
        self,
        segments: Sequence[str],
        type_label: str | None,
        description: str,
        constraints: list[str],
        schema: dict[str, Any],
    ) -> list[str]:
        """Build all descriptive sections shown inside the card body.

        Each section is returned as an HTML string. The caller can then drop the
        sections into the card body in whatever order they choose.
        """
        sections = [self._breadcrumb(segments), self._meta_list(type_label)]
        if description:
            sections.insert(
                1,
                f'<div class="schema-card-description">{html.escape(description)}</div>',
            )
        if constraints:
            sections.append(self._bullet_section("Constraints", constraints))
        examples = schema.get("examples")
        if isinstance(examples, Sequence) and examples:
            formatted = ", ".join(format_literal(example) for example in examples)
            sections.append(
                '<div class="schema-card-section"><div class="schema-card-section-title">Examples</div>'
                f'<div class="schema-card-examples">{formatted}</div></div>'
            )
        return sections

    def build_body_html(
        self, sections: Sequence[str], children_html: str, summary_only_inline: bool
    ) -> str:
        """Return the final body HTML block unless the card is summary-only."""
        if summary_only_inline:
            return ""
        body = self._compose_body_content(sections, children_html)
        return self._wrap_body_content(body)

    def build_summary(
        self,
        *,
        title: str,
        summary_desc: str,
        badges: Sequence[str],
        inline: bool,
        allow_inline_title: bool,
    ) -> tuple[str, bool]:
        """Compose the summary block (title/description + badges).

        Returns both the HTML string as well as a flag indicating whether the
        summary contains only badges (used to tweak CSS classes later).
        """
        title_span = self._summary_title_span(title, inline, allow_inline_title)
        summary_main = self._summary_main_block(title_span, summary_desc)
        summary_badge_only = not summary_main
        summary_inner = self._summary_wrapper(summary_main, badges, summary_badge_only)
        return summary_inner, summary_badge_only

    def render_inline_card(
        self, anchor: str, summary_inner: str, body_html: str, permalink_html: str
    ) -> str:
        """Render a non-collapsible inline card (used for scalar array items).

        Inline cards do not use <details>. Instead we simply stack the summary
        and body markup and wrap them in a div.
        """
        summary_block = self._append_permalink(summary_inner, permalink_html)
        card_inner = self._card_inner(summary_block, body_html)
        classes = [
            "schema-card",
            "schema-card--inline",
            "schema-card--inline-static",
        ]
        if not body_html:
            classes.append("schema-card--summary-only")
        return self._wrap_card_div(anchor, classes, card_inner)

    def render_block_card(
        self,
        anchor: str,
        summary_inner: str,
        body_html: str,
        summary_badge_only: bool,
        open_card: bool,
        permalink_html: str,
    ) -> str:
        """Render a standard collapsible card (summary + optional permalink + body)."""
        summary_attr = self._summary_toggle_attr(summary_badge_only)
        # The summary element doubles as DETAILS/summary toggle in HTML.
        header_html = f"<summary{summary_attr}>\n{summary_inner}\n</summary>"
        if permalink_html:
            header_html = f"{header_html}\n{permalink_html}"
        card_inner = self._block_card_inner(header_html, body_html)
        return self._wrap_details_card(anchor, card_inner, open_card)

    def render_children_container(self, child_sections: Sequence[str]) -> str:
        """Render the wrapper for nested child cards."""
        return (
            '<div class="schema-card-children">'
            + '<div class="schema-card-section-title">Nested fields</div>'
            + "".join(child_sections)
            + "</div>"
        )

    def _breadcrumb(self, segments: Sequence[str]) -> str:
        """Render clickable breadcrumbs showing the nested path."""
        crumbs = []
        for idx, segment in enumerate(segments):
            if segment == "[item]":
                continue
            label = (self.root_title or "root") if idx == 0 else self._segment_label(segment)
            prefix = segments[: idx + 1]
            anchor = self.anchor(prefix)
            crumbs.append(f'<a href="#{anchor}">{html.escape(label)}</a>')
        return (
            '<div class="schema-card-path">'
            + " › ".join(crumbs)
            + "</div>"
        )

    @staticmethod
    def _wrap_badge_container(badge: str) -> str:
        """Wrap a badge span for styling while keeping it non-interactive."""
        return f'<span class="schema-card-badge-link" role="presentation">{badge}</span>'

    @staticmethod
    def _meta_list(type_label: str | None) -> str:
        """Render the basic metadata definition list."""
        if not type_label:
            return ""
        return (
            '<dl class="schema-card-meta">\n'
            f"  <div><dt>Type</dt><dd>{html.escape(type_label)}</dd></div>\n"
            "</dl>"
        )

    @staticmethod
    def _bullet_section(title: str, items: Sequence[str]) -> str:
        """Render a bullet list section block."""
        lis = "".join(f"<li>{item}</li>" for item in items)
        return (
            '<div class="schema-card-section">\n'
            f'  <div class="schema-card-section-title">{html.escape(title)}</div>\n'
            f'  <ul class="schema-card-list">{lis}</ul>\n'
            "</div>"
        )

    @staticmethod
    def _segment_label(segment: str) -> str:
        """Human-friendly label for breadcrumb segments."""
        if segment == "[*]":
            return "additional"
        return segment

    @staticmethod
    def _slug(name: str) -> str:
        """Convert a string into a slug suitable for anchors."""
        return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")

    @staticmethod
    def _compose_body_content(sections: Sequence[str], children_html: str) -> str:
        """Join section blocks with any child HTML."""
        # Children HTML is already fully wrapped so we can simply concatenate it.
        return "\n".join(filter(None, sections)) + children_html

    @staticmethod
    def _wrap_body_content(body: str) -> str:
        """Wrap the composed body content with the body container."""
        return (
            '  <div class="schema-card-body">\n'
            f"    {body}\n"
            "  </div>"
        )

    @staticmethod
    def _card_inner(summary_inner: str, body_html: str) -> str:
        """Concatenate summary and body parts for inline cards."""
        parts = [summary_inner]
        if body_html:
            parts.append(body_html)
        return "\n".join(part for part in parts if part)

    @staticmethod
    def _wrap_card_div(anchor: str, classes: Sequence[str], inner: str) -> str:
        """Wrap inline cards in a div with the provided classes."""
        return (
            f'<div class="{" ".join(classes)}" id="{anchor}">\n'
            f"  {inner}\n"
            "</div>"
        )

    @staticmethod
    def _append_permalink(summary_inner: str, permalink_html: str) -> str:
        """Append a permalink control outside the summary content."""
        if not permalink_html:
            return summary_inner
        return f"{summary_inner}\n{permalink_html}"

    @staticmethod
    def permalink_link(anchor: str, title: str) -> str:
        """Return a permalink anchor placed outside <summary> for accessibility."""
        safe_title = html.escape(title)
        return (
            f'<a class="schema-card-anchor-link" href="#{anchor}" '
            f'aria-label="Permalink to {safe_title}">#</a>'
        )

    @staticmethod
    def _block_card_inner(header_html: str, body_html: str) -> str:
        """Assemble the inner contents of a block card."""
        return (
            f"{header_html}\n"
            f"  {body_html}"
        ).strip()

    @staticmethod
    def _wrap_details_card(anchor: str, card_inner: str, open_card: bool) -> str:
        """Wrap block cards inside a <details> element."""
        open_attr = " open" if open_card else ""
        return (
            f'<details class="schema-card"{open_attr} id="{anchor}">\n'
            f"  {card_inner}\n"
            "</details>"
        )

    @staticmethod
    def _summary_title_span(title: str, inline: bool, allow_inline_title: bool) -> str:
        """Return the summary title span when it should be shown."""
        if title and (not inline or allow_inline_title):
            return (
                '<span class="schema-card-title" style="-webkit-font-feature-settings:\'tnum\' 0,\'lnum\' 0;'
                " font-feature-settings:'tnum' 0,'lnum' 0;\">"
                f"{html.escape(title)}</span>"
            )
        return ""

    @staticmethod
    def _summary_main_block(title_span: str, summary_desc: str) -> str:
        """Compose the inner summary block containing title and description."""
        if not (title_span or summary_desc):
            # Returning empty string keeps the downstream wrapper logic simple.
            return ""
        return "\n".join(
            [
                '    <div class="schema-card-summary-main">',
                f"      {title_span}",
                f"      {summary_desc}",
                "    </div>",
            ]
        ).rstrip()

    @staticmethod
    def _summary_toggle_attr(summary_badge_only: bool) -> str:
        """Return the class attribute for the summary toggle area."""
        if summary_badge_only:
            # Badge-only summaries need a modifier to tighten spacing in CSS.
            return ' class="schema-card-summary-toggle schema-card-summary-toggle--badges-only"'
        return ' class="schema-card-summary-toggle"'

    @staticmethod
    def _summary_wrapper(
        summary_main: str, badges: Sequence[str], summary_badge_only: bool
    ) -> str:
        """Wrap the summary main content and badges."""
        summary_classes = ["schema-card-summary"]
        if summary_badge_only:
            summary_classes.append("schema-card-summary--badges-only")
        return (
            f"  <div class=\"{' '.join(summary_classes)}\">\n"
            f"    {summary_main}\n"
            f'    <div class="schema-card-summary-badges">{"".join(badges)}</div>\n'
            "  </div>"
        ).strip()
