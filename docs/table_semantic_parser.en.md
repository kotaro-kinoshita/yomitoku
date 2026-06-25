# Using the Table Semantic Parser

`TableSemanticParser` is a module that detects **tables** from form/document images and **converts them into structured data**. It also automatically estimates and analyzes semantic structures such as **cell roles** (e.g., `header` / `cell`) and **Key-Value pairs (label → value)**.

**Key features**

* Table region detection
* Cell segmentation (estimating cell boundaries)
* OCR (reading text inside cells)
* Cell role estimation (header / value)
* Key-Value estimation (data represented as 1:1 relationships between keys and values)
* Grid structure estimation (matrix-like, grid-shaped data)
* Search/export per table ID (`t0`, `t1`, ...)

All of these processes are performed end-to-end in a single pipeline.

In the following sections, we explain the functionality with source code and examples.

> Note: Only forms that **have separator lines (ruled lines)** are supported. Tables without separator lines are not supported.

---

## Sample Data

![Sample Data](assets/table.jpg)

## Sample Code

<!--codeinclude-->

[demo/simple_table_semantic_analysis.py](../demo/simple_table_semantic_analysis.py)

<!--/codeinclude-->

---

## Output of the Table Semantic Parser

See the [schema](schemas.en.md) for details.

* `paragraphs`: Information about textual components such as paragraphs and titles in the document. It includes text strings and layout-related text information.
* `tables`: A table region within the document. It contains information such as `cells`, `kv_items`, and `grids`.

  * `cells`: Holds each cell in the table, including the text content inside the cell and the cell role (e.g., `header` or `cell`).
  * `kv_items`, `grids`: Semantically structured information derived from the table. Details are explained later.
* `words`: Information about text in the document. This is obtained from the text detection model and includes positions and the recognized text within the regions.

## Visualization of the Results

![Example of Visualization](assets/vis_cell.jpg)

---

## Two Types of Structured Outputs: `kv_items` and `grids`

The Table Semantic Parser not only reconstructs a table as a “set of cells,” but also estimates and outputs the **semantic structure** of the table. The key concepts are **`kv_items`** and **`grids`**.

### `kv_items` (Key-Value)

`kv_items` represent information in a table where a **header (label)** and a **value** correspond **1:1**.

For example, in the sample form, the following items correspond to `kv_items` (highlighted with red boxes in the visualization):

* `利用目的` → `セミナー`
* `施設名称` → `MLism株式会社`
* `実施内容` → `YomiTokuの利用方法に関する説明会`

In other words, `kv_items` are defined as data where a label and a value are semantically paired and it is natural to treat them as a dictionary (Key-Value).

### `grids` (Grid / Tabular Records)

`grids` represent data that can be interpreted as a **row/column grid structure**, i.e., data that can be interpreted as a **matrix** (highlighted with blue boxes in the visualization).

For example, the “利用希望日” section in the sample form can be expressed as repeating rows with columns such as `日付`, `入室時刻`, and `退室時刻`, and thus corresponds to a `grid`:

* `2025年01月30日(月曜日)` / `10時00分` / `17時00分`
* `2025年02月1日(火曜日)` / `10時00分` / `17時00分`
* ...

In other words, `grids` are defined as grid-shaped data where **multiple rows exist as records** and **columns are defined as attributes**.

### Cells

Cells inside each table are stored in a schema called `cells`. Each cell is assigned a table-unique identifier `id` (e.g., `c0`).

`kv_items` and `grids` store cell IDs, and by referencing these IDs you can access detailed cell information. In addition to using structured data such as `kv_items` and `grids`, APIs are also provided to reference cells by `id` or to retrieve specific cell information based on positional relationships.

---

## Python API

The analysis results produced by the Table Semantic Parser can be accessed via APIs to retrieve information by searching IDs, strings, or keys, and to convert/export the data into structured formats.

Some APIs apply to the entire document, while others apply to a specific table. For clarity, functions applied to the whole document are prefixed with `results.`, and functions applied at the table level are prefixed with `table.` in this documentation.

To apply processing to a specific table, first obtain the table object using one of the following two functions.

There are two ways to specify a table: by ID or by position. After obtaining the table, you can reference cells or use semantic structures to extract specific information.

---

### `results.find_table_by_id`

Retrieve a specific table by ID. The `table_id` can be identified from the JSON output or the visualization. Since sorting and ID assignment follow a deterministic order based on positions, the same format is likely to yield the same ID. However, IDs may change due to false positives or missed detections.

**Arguments**

* `table_id: str`: The table ID to retrieve

**Returns**

* `TableSemanticContentsSchema`: Holds structured information for the specified table.

**Example**

```python
table = results.find_table_by_id("t0")
```

---

### `results.find_table_by_position`

Retrieve table information that overlaps with the specified region. This is useful when ID-based selection is unstable due to false positives or missed detections.

**Arguments**

* `box: List[int]`: `[x1, y1, x2, y2]`
  The rectangle is defined by two points: the top-left `(x1, y1)` and the bottom-right `(x2, y2)`.
  `x1, y1` are the coordinates of the top-left corner, and `x2, y2` are the coordinates of the bottom-right corner.

**Returns**

* `TableSemanticContentsSchema`: Holds structured information for the specified table.

**Example**

```python
table = results.find_table_by_position([149, 498, 1503, 1293])
```

---

## Retrieving Information by Position and Cells (ID / Position / Neighborhood)

### A. Search the Entire Document

#### `results.search_words_by_position`

Retrieve `words` contained within the specified region to check OCR text. This is useful when you already know that the target text is located in a specific area.

**Arguments**

* `box: List[int]`: `[x1, y1, x2, y2]`
  The rectangle is defined by the top-left `(x1, y1)` and bottom-right `(x2, y2)` points.

**Returns**

* `str`: The text string contained in the region

**Example**

```python
word = results.search_words_by_position([365, 1444, 1498, 1539])
print(word)
```

**Output**

```
MLism 株式会社
```

---

### B. Search Within a Specific Table (`table` object)

#### `table.find_cell_by_id`

Retrieve a cell in the table by `cell_id`. Since sorting and ID assignment follow a deterministic order based on positions, the same format is likely to yield the same ID. However, IDs may change due to false positives or missed detections.

If IDs change, you may not be able to extract information correctly. In such cases, try extraction methods based on relative positions or semantic structures.

**Arguments**

* `cell_id: str`: Cell ID (e.g., `"c0"`)

**Returns**

* `CellSchema`: Contains detailed cell information such as ID, text, bounding box, and row/column indices.

**Example**

```python
cell = table.find_cell_by_id("c0")
print(cell.id, cell.contents, cell.box)
```

**Output**

```
c0 利用情報 [149, 498, 1501, 550]
```

---

#### `table.search_cells_by_query`

List cells whose text content partially matches the given query.

**Arguments**

* `query: str`: Text to search for

**Returns**

* `List[CellSchema]`: A list of matching cells

**Example**

```python
cells = table.search_cells_by_query("2025")
for c in cells:
    print(c.id, c.contents, c.box)
```

**Output**

```
c11 2025年 01月 30日(月 曜日) [365, 888, 946, 968]
c14 2025年 02月 1日(火 曜日) [365, 968, 946, 1049]
```

**Use cases**

* Extract cells that contain a specific string
* Retrieve the position of cells containing a specific string

---

#### `table.search_cells_right_of_key_text` / `table.search_cells_below_key_text`

Retrieve neighboring cells to the right (or below) the cell containing the given key string.

Many Japanese forms have a specific key and store the information to retrieve at a relative position from that key. This API uses that property to get neighboring cell information to the right or below. Key matching is done by partial match.

**Arguments**

* `key: str` (partial match)

**Returns**

* `List[CellSchema]`: A list of matching neighboring cells

**Example**

```python
# Retrieve the neighboring cell(s) to the right of the cell containing "実施内容"
vals = table.search_cells_right_of_key_text("実施内容")
for v in vals:
    print(v.id, v.contents)

# Retrieve the cell(s) directly below the cell containing "実施内容"
vals = table.search_cells_below_key_text("実施内容")
for v in vals:
    print(v.id, v.contents)
```

**Output**

```
c6 YomiToku の利用方法に関する説明会
c7 利用希望日
```

---

## Extracting Information from Semantic Structures (`kv_items` / `grids`)

Semantic-structure-based retrieval uses **key-to-value mapping (`kv_items`)** and **matrix structures (`grids`)**. This approach is robust to minor format variations in documents. Since structuring is estimated automatically, it may fail in some cases. If extraction does not work well, try using cell IDs or relative-position-based methods.

---

### A. Search the Entire Document (No Table Specified)

#### `results.search_kv_items_by_key`

Search `kv_items` across the entire analysis result (possibly including multiple tables) by partial matching on the key.

**Arguments**

* `key: str`: Search key (partial match)

**Returns**

* `List[dict]`: Each element is `{"key": [CellSchema, ...], "value": [CellSchema, ...]}`

**Example**

```python
kv_items = results.search_kv_items_by_key(key="団体名")
for kv in kv_items:
    keys = "_".join([k.contents for k in kv["key"]])
    val = kv["value"].contents
    print(keys, val)
```

**Output**

```
団 体 名 エムリズムカブシキガイシャ
団 体 名 MLism 株式会社
```

> Note
> If cells have nested structures, the key may consist of multiple cells, so `key` is returned as a list.
> Multiple values may also exist depending on the search, so results are returned as a list.

---

### B. Search Within a Specific Table (`table` object)

#### `table.search_kv_items_by_key`

Search for kv_items within table `t0` only by partial matching on the key.

**Arguments**

* `key: str`

**Returns**

* `List[dict]`: `{"key": [CellSchema], "value": [CellSchema]}`

**Example**

```python
hits = table.search_kv_items_by_key(key="利用目的")
print(hits)
```

**Output**

```python
[
    {
        'key': [
            CellSchema(
                meta={},
                contents='利 用 目 的',
                role='header',
                id='c3',
                box=[149, 645, 364, 741],
                row=None,
                col=None,
                row_span=None,
                col_span=None
            )
        ], 
        'value': CellSchema(
            meta={}, 
            contents='セミナー', 
            role='cell', 
            id='c4', 
            box=[365, 645, 1498, 741], 
            row=None, 
            col=None, 
            row_span=None, 
            col_span=None
        )
    }
]
```

---

#### `table.view.kv_items_to_dict`

Expand cell references in `kv_items` and convert them into a dictionary (`key string → value string`).

**Returns**

* `dict[str, str]`

**Example**

```python
kv = table.view.kv_items_to_dict()
print(kv)
print(kv.get("利用目的"))
```

**Output**

```
{
  "施設名称": "MLism株式会社",
  "利用目的": "セミナー",
  "実施内容": "YomiTokuの利用方法に関する説明会"
}
セミナー
```

---

#### `table.view.grids_to_dict()`

Convert structured `grid` information into a list of record dictionaries.

**Returns**

* `list[dict]`: `[{ "id": grid_id, "rows": [ {...}, ...]}]`

**Example**

```python
grids = table.view.grids_to_dict()
for g in grids:
    print("grid:", g["id"])
    for row in g["rows"]:
        print(row)
```

**Output**

```
grid: g0
{'日付': '2025年01月30日(月曜日)', '入室時刻': '10時00分', '退室時刻': '17時00分'}
{'日付': '2025年02月1日(火曜日)', '入室時刻': '10時00分', '退室時刻': '17時00分'}
{'日付': '年月日(曜日)', '入室時刻': '時分', '退室時刻': '時分'}
{'日付': '年月日(曜日)', '入室時刻': '時分', '退室時刻': '時分'}
{'日付': '年月日(曜日)', '入室時刻': '時分', '退室時刻': '時分'}
```

---

## Exporting Structured Data (JSON / CSV)

You can export the semantic analysis results into JSON or CSV for integration with other systems. You can export data for the entire document or for a specific table.

### A. Export the Entire Document

#### `results.to_json`

Save the analysis result of the entire document as JSON.

**Arguments**

* `out_path: str`: Path to the output JSON file

**Example**

```python
analyzer = TableSemanticParser(
    device="cuda",
    visualize=True,
)
imgs = load_pdf(path_img)
results, vis_layout, vis_ocr = analyzer(imgs[0])

# Save the document analysis result as JSON
results.to_json("out/result.json")
```

---

#### `TableSemanticParserSchema.load_json`

`load_json` is a utility that reads a saved JSON result and restores it as `TableSemanticParserSchema` (= `results`). This enables re-running extraction logic without reprocessing with AI. Page-count billing (or processing count) is performed only when AI is invoked. Therefore, running extraction on results loaded via `load_json` does not count toward processing.

During development/validation, or when you want to reduce processing time, it is recommended to save once with `to_json` and then reload with `load_json`.

**Arguments**

* `json_path: str`: Path to the JSON file

**Example**

```python
from yomitoku.schemas.table_semantic_parser import TableSemanticParserSchema

json_path = "out/result.json"
results = TableSemanticParserSchema.load_json(json_path)
```

---

#### `results.to_dict`

`to_dict` is a utility method that aggregates, for every table in the document, the following by table ID:

* `kv_items` (Key-Value structures)
* `grids` (matrix structures)

It converts the result into a Python `dict` (structured data). It reduces information such as cell bounding boxes and converts into structured data composed primarily of text.

This is suitable when you want to use the parsed results directly in application logic or API responses, or when you want to feed structured document information into a generative AI model.

**Example**

```python
from pprint import pprint
parsed = results.to_dict()
pprint(parsed)
```

**Output**

```python
{
    't0': {
        'grids': [
            {
                'id': 'g0',
                'rows': [
                    {'入室時刻': '10時00分','日付': '2025年01月30日(月曜日)','退室時刻': '17時00分'},
                    {'入室時刻': '10時00分','日付': '2025年02月1日(火曜日)','退室時刻': '17時00分'},
                    {'入室時刻': '時分', '日付': '年月日(曜日)', '退室時刻': '時分'},
                    {'入室時刻': '時分', '日付': '年月日(曜日)', '退室時刻': '時分'},
                    {'入室時刻': '時分', '日付': '年月日(曜日)', '退室時刻': '時分'}]
            }
        ],
        'kv_items': {
            '利用目的': 'セミナー',
            '実施内容': 'YomiTokuの利用方法に関する説明会',
            '施設名称': 'MLism株式会社'
        }
    },
    't1': {
        'grids': [],
        'kv_items': {
            '代表者_0': 'キノシタコウタロウ',
            '代表者_1': '木之下滉大郎',
            '住所': '〒277-8520千葉県柏市若柴178番地4柏の葉キャンパス148街区2ショップ&オフィス棟6F',
            '団体名_0': 'エムリズムカブシキガイシャ',
            '団体名_1': 'MLism株式会社',
            '電話番号': '090-1234-5678'
        }
    }
}
```

---

### B. Export Per Table

#### `table.kv_items_to_json`

Export `kv_items` in the table into a single JSON file.

**Arguments**

* `out_path: str`

**Returns**

* `dict` (In addition to writing a file, it also returns the dictionary in memory.)

**Example**

```python
table = results.find_table_by_id("t0")
kv = table.export.kv_items_to_json("out/kv_items.json")
print(kv)
```

**Output**

```json
{
    "施設名称": "MLism株式会社",
    "利用目的": "セミナー",
    "実施内容": "YomiTokuの利用方法に関する説明会"
}
```

---

#### `table.grids_to_json`

Export `grids` in the table into a single JSON file. Since a table may contain multiple grids, the output is a list.

**Arguments**

* `out_path: str`

**Returns**

* `list[dict]`: Dictionary data converted from grids into record rows

**Example**

```python
table = results.find_table_by_id("t0")
grids = table.export.grids_to_json("out/grids.json")
print(grids)
```

**Output**

```json
[
    {
        "id": "g0",
        "rows": [
            {
                "日付": "2025年01月30日(月曜日)",
                "入室時刻": "10時00分",
                "退室時刻": "17時00分"
            },
            {
                "日付": "2025年02月1日(火曜日)",
                "入室時刻": "10時00分",
                "退室時刻": "17時00分"
            },
            {
                "日付": "年月日(曜日)",
                "入室時刻": "時分",
                "退室時刻": "時分"
            },
            {
                "日付": "年月日(曜日)",
                "入室時刻": "時分",
                "退室時刻": "時分"
            },
            {
                "日付": "年月日(曜日)",
                "入室時刻": "時分",
                "退室時刻": "時分"
            }
        ]
    }
]
```

---

#### `table.grids_to_csv`

Export `grids` in the table as CSV. You can optionally specify column names to export only certain fields.

**Arguments**

* `out_path: str`: Path to the output CSV file
* `columns: List[str] | None = None`: Column names to export (matched by partial match)
* `ignore_space: bool = True`: Whether to remove spaces when exporting

**Returns**

* `List[List[str]]` (2D array representing CSV rows)

In the example below, columns are specified. If columns are not specified, all information in the grid is exported.

**Example**

```python
table = results.find_table_by_id("t0")
csvs = table.export.grids_to_csv("out/grid.csv", columns=["入室", "退室"])
```

**Output**
Exporting only “入室” and “退室” as columns:

```
入室時刻,退室時刻
10時00分,17時00分
10時00分,17時00分
時分,時分
時分,時分
時分,時分
```

**Generated file names**

* A suffix with the `grid_id` is added, e.g. `out/grid_0.csv`.
  If multiple grids exist in the table, the suffix changes accordingly, e.g. `out/grid_1.csv`.


