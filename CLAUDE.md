# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YomiToku is an AI-powered document image analysis engine (Document AI) specialized for Japanese. It provides full OCR and layout analysis capabilities — recognizing, extracting, and converting text and diagrams from images. License: CC BY-NC-SA 4.0.

## Common Commands

```bash
# Install (editable)
uv pip install -e .

# Lint & Format
tox -e lint              # Run ruff linter via tox
ruff check --fix         # Auto-fix lint issues directly
ruff format              # Format code

# Test
pytest tests/                         # All tests
pytest tests/test_ocr.py              # Single file
pytest tests/test_ocr.py::test_name   # Single test
tox -e py312                          # Via tox for specific Python version

# Documentation
mkdocs serve             # Local preview at localhost:8000

# System Dependencies (Ubuntu)
sudo apt install libopencv-dev poppler-utils
```

## Architecture

### Processing Pipeline

`DocumentAnalyzer` (`document_analyzer.py`) is the top-level orchestrator. It runs components concurrently via `ThreadPoolExecutor`:

1. **RotateDetector** (optional) — corrects page rotation
2. **OCR** (`ocr.py`) — combines `TextDetector` (DBNet+) + `TextRecognizer` (PARSeq)
3. **LayoutAnalyzer** (`layout_analyzer.py`) — combines `LayoutParser` (RTDETRv2) + `TableStructureRecognizer` (RTDETRv2)
4. **Reading order estimation** (`reading_order.py`) — DFS-based graph algorithm
5. **Export** — JSON, CSV, HTML, Markdown, or searchable PDF

Each component can be used independently (e.g., `OCR` alone for text extraction).

### Key Design Patterns

- **Model catalog**: Each module uses a `BaseModelCatalog` subclass (in `base.py`) that maps model name strings to `(config, model_class)` pairs. New model variants are registered in the catalog.
- **Config-driven**: OmegaConf dataclass configs in `src/yomitoku/configs/` define model architecture, HuggingFace Hub repo paths, and processing parameters. Models auto-download from HF Hub on first use. Configs can be overridden with YAML files via `load_config()`.
- **Pydantic schemas**: All outputs are validated Pydantic models in `src/yomitoku/schemas/` with built-in `.to_json()`, `.to_html()`, `.to_csv()`, `.to_markdown()` export methods.
- **Dual inference**: All modules support PyTorch and ONNX runtime, toggled via `infer_onnx` flag. ONNX models auto-convert and cache on first use.
- **Observer pattern**: `observer` decorator on `BaseModule.__call__` provides timing and error logging.

### Source Layout (`src/yomitoku/`)

| Directory/File | Purpose |
|---|---|
| `document_analyzer.py` | Top-level pipeline orchestrator |
| `ocr.py` | OCR = TextDetector + TextRecognizer |
| `layout_analyzer.py` | LayoutParser + TableStructureRecognizer |
| `reading_order.py` | Reading order estimation |
| `base.py` | BaseModule, BaseModelCatalog, config/model loading |
| `text_detector.py` | DBNet+ text detection |
| `text_recognizer.py` | PARSeq text recognition |
| `layout_parser.py` | RTDETRv2 layout analysis |
| `table_structure_recognizer.py` | RTDETRv2 table structure |
| `table_cell_detector.py` | Cell-level table detection |
| `table_semantic_parser.py` | Advanced semantic table parsing |
| `grid_parser.py` | Grid structure parsing |
| `kv_parser.py` | Key-Value pair extraction |
| `models/` | Neural network definitions (DBNet, PARSeq, RTDETRv2) |
| `configs/` | OmegaConf config dataclasses per model variant |
| `schemas/` | Pydantic output schemas |
| `postprocessor/` | Model output post-processing |
| `export/` | Output formatters |
| `extractor/` | Structured data extraction (rule-based and LLM-based) |
| `data/` | Image/PDF loading (`load_image`, `load_pdf`) |
| `utils/` | Logging, visualization, graph algorithms, searchable PDF |
| `cli/` | CLI entry points |
| `resource/` | Fonts, charset files |

### CLI Entry Points

| Command | Module | Purpose |
|---|---|---|
| `yomitoku` | `cli.main` | Main document analysis |
| `yomitoku_mcp` | `cli.mcp_server` | MCP server |
| `download_model` | `cli.download_model` | Model downloader |
| `yomitoku_extract` | `cli.extract_rule` | Rule-based structured data extraction |
| `yomitoku_extract_with_llm` | `cli.extract` | LLM-based structured data extraction |

### Available Models

| Component | Variants |
|---|---|
| TextDetector | `dbnet`, `dbnetv2` |
| TextRecognizer | `parseq`, `parseqv2`, `parseq-small`, `parseq-tiny` |
| LayoutParser | `layout_parser_rtdtrv2`, `layout_parser_rtdtrv2_v2` |
| TableStructureRecognizer | `table_structure_recognizer_rtdtrv2` |
| TableCellDetector | `table_cell_parser_rtdtrv2_beta` |

Lite mode (`--lite`) uses `parseq-small`/`parseq-tiny` for CPU-friendly inference.

## Conventions

- Python `>=3.10,<3.14`
- Package manager: `uv` with `tox-uv` for CI
- Build: hatchling + uv-dynamic-versioning (version from git semver tags)
- Linting/formatting: `ruff` via pre-commit hooks (hooks use `.tox/lint/bin/ruff`)
- Input formats: JPG, JPEG, PNG, BMP, TIFF, PDF
- Output formats: JSON, CSV, HTML, Markdown, searchable PDF
- Optional extras: `mcp` (MCP server support), `extract` (LLM extraction with OpenAI)
