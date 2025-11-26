"""Custom macros to render JSON schemas as nested cards within MkDocs."""

from macros.schema_renderer import SchemaRenderer


def define_env(env) -> None:
    """Entry point used by mkdocs-macros-plugin."""

    # Reuse a single renderer per build so schema caches stay warm.
    renderer = SchemaRenderer()

    @env.macro
    def schema_cards(schema_name: str) -> str:
        """Render the schema with the given base filename (without extension)."""
        return renderer.render(schema_name)
