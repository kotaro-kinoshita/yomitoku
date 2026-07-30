# CLI Usage (Table Semantic Parser)

The `yomitoku_table` command analyzes whole documents with the Table Semantic Parser and exports the semantic structure of tables (key-value items and grids) together with paragraphs as per-page structured JSON.

Model weights are downloaded from the Hugging Face Hub on the first run.

```bash
yomitoku_table ${path_data} -o results -v
```

| Option | Description |
| :-- | :-- |
| `${path_data}` | Path to a target image/PDF file or a directory containing them. Directories are processed recursively. |
| `-o`, `--outdir` | Output directory (created if missing). Default: `results` |
| `-v`, `--vis` | Export visualization images (`*_layout.jpg` / `*_ocr.jpg`). |
| `-l`, `--lite` | Use lite models (for CPU). |
| `-d`, `--device` | Device to run the models on (cuda \| cpu \| mps). Default: `cuda` |
| `--raw` | Output the normalized `TableSemanticParserSchema` JSON. |
| `--simple` | Output text-only structured JSON without coordinates and other metadata. |
| `--cell_name` | Table cell detector model. Default: `rtdetrv2` |
| `--cell_cfg` | Path to a config file (YAML) for the cell detector. |
| `--lp_name` / `--lp_cfg` | Layout parser (table detector) model name / config file. Default: `rtdetrv2v2` |
| `--td_name` / `--td_cfg` | Text detector model name / config file. Default: `dbnetv2_1` |
| `--tr_name` / `--tr_cfg` | Text recognizer model name / config file. Default: `parseq-large-v4_1` |
| `--template` | Apply a table template JSON (skips grid/kv inference). |
| `--grid_only` | Parse only grid regions (skip key-value items). |
| `--kv_only` | Parse only key-value items (skip grids). |
| `--pages` | Pages to process (e.g. `1,2,5-10`, 1-indexed). Default: all pages |
| `--dpi` | Resolution for loading PDFs. Default: `200` |
| `--encoding` | Output file encoding (`utf-8` \| `utf-8-sig` \| `shift-jis` \| `euc-jp` \| `cp932`). |

Results are saved per page as `{stem}_p{page}.json`.

## Output Formats

### Default (structured JSON)

Cell ids in `kv_items` / `grids` are resolved into text, and the originating cell ids and coordinates are embedded as `key_cells` / `value_cells`.

- When multiple values are associated with the same key cells, the values are joined in spatial order (vertical/horizontal is auto-detected) with a line break, and `value_cells` lists the source cells in the same order.
- Merging is decided by the key cell ids, not the key text, so distinct fields that happen to share the same label text are never merged.
- Standalone cells without a key (empty `key`) also remain separate entries.

```bash
yomitoku_table ${path_data} -o results
```

```json
{
    "tables": [
        {
            "id": "t0",
            "box": [150, 500, 1500, 840],
            "style": "border",
            "kv_items": [
                {
                    "key": "Facility name",
                    "value": "MLism Inc.",
                    "key_cells": [{"id": "c1", "box": [150, 550, 365, 645]}],
                    "value_cells": [{"id": "c2", "box": [365, 550, 1499, 645]}]
                }
            ],
            "grids": [
                {
                    "id": "g0",
                    "box": [150, 840, 1500, 1370],
                    "n_row": 6,
                    "n_col": 4,
                    "rows": [
                        {
                            "cells": [
                                {
                                    "key": "Date",
                                    "value": "2025-01-30 (Mon)",
                                    "key_cells": [{"id": "c7", "box": [365, 840, 947, 888]}],
                                    "value_cells": [{"id": "c11", "box": [365, 888, 947, 968]}]
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    ],
    "paragraphs": [
        {
            "id": "p0",
            "box": [669, 226, 983, 274],
            "score": 0.97,
            "role": "section_headings",
            "contents": "Facility Use Application"
        }
    ]
}
```

The same view is available from the Python API via `results.to_structured()`.

### `--simple` (text only)

Outputs a text-only form without coordinates or cell references. `kv_items` becomes a `{key: value}` mapping, grid rows become `{column header: value}` mappings, and paragraphs become plain strings. Duplicate key texts are distinguished with `_0` / `_1` suffixes.

```bash
yomitoku_table ${path_data} -o results --simple
```

The same view is available via `results.to_simple()`.

### `--raw` (normalized schema)

Outputs the `TableSemanticParserSchema` as is: a lossless format containing `cells` (a dict keyed by cell id), `kv_items` (cell id references), `grids`, and `words`. Use this for template round-trips and re-analysis. See [Table Semantic Parser](table_semantic_parser.en.md) for details.

```bash
yomitoku_table ${path_data} -o results --raw
```

## Lite Mode

With `--lite`, lightweight models are used for faster inference on CPU, at the cost of some recognition accuracy.

```bash
yomitoku_table ${path_data} --lite -d cpu
```

## Specifying Models and Configs

Model names and config files (YAML) can be specified per module.

```bash
yomitoku_table ${path_data} \
  --cell_name rtdetrv2 \
  --tr_name parseq-small \
  --td_cfg text_detector.yaml
```

| Module | Name option | Config option | Choices |
| :-- | :-- | :-- | :-- |
| Cell detector | `--cell_name` | `--cell_cfg` | `rtdetrv2` (official, 960 input) |
| Table detector | `--lp_name` | `--lp_cfg` | `rtdetrv2`, `rtdetrv2v2` |
| Text detector | `--td_name` | `--td_cfg` | `dbnet`, `dbnetv2`, `dbnetv2_1` |
| Text recognizer | `--tr_name` | `--tr_cfg` | `parseq`, `parseqv2`, `parseq-small`, `parseq-tiny`, `parseq-large-v4_1` |

## Applying a Template

With `--template`, grid/kv inference is skipped and the definitions in the template JSON are applied instead. Templates can be created from `--raw` output via `save_template_json()`.

```bash
yomitoku_table ${path_data} --template template.json
```

## Parsing Only Grids / Key-Values

```bash
# grids only
yomitoku_table ${path_data} --grid_only

# key-value items only
yomitoku_table ${path_data} --kv_only
```

## Selecting Pages

Specify pages to process with `--pages` (1-indexed, comma-separated, ranges allowed).

```bash
yomitoku_table ${path_data} --pages 1,3-5
```

## Visualization

With `-v`, the following images are exported per page:

- `*_layout.jpg`: tables, paragraphs, and cell roles (green = header, blue = cell, magenta = empty). Resolved key-value links are drawn as **green arrows** and grid structures as **blue boxes and arrows**.
- `*_ocr.jpg`: text detection/recognition results.

```bash
yomitoku_table ${path_data} -o results -v
```
