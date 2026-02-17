# Yomitoku Extractor

Yomitoku Extractor is a tool for extracting structured data from document images and PDFs according to a schema definition.
It uses YomiToku's table analysis engine (TableSemanticParser) for OCR and layout analysis, then extracts field values based on a YAML-defined schema.

Two extraction modes are available:

| Command | Mode | Features |
| :--- | :--- | :--- |
| `yomitoku_extract` | Rule-based | No LLM required. Fast extraction via KV search and grid matching |
| `yomitoku_extract_with_llm` | LLM-based | More flexible extraction using an LLM server such as vLLM |

### Choosing the Right Mode

Choose between the two extraction modes based on your document type and extraction requirements.

| Aspect | `yomitoku_extract` (Rule-based) | `yomitoku_extract_with_llm` (LLM-based) |
| :--- | :--- | :--- |
| **Best for** | Fixed-format documents (application forms, reports, slips) | Variable-format documents (business cards, receipts, invoices) |
| **Layout** | Layout is consistent — same items appear in the same positions | Layout varies by issuer or vendor |
| **Target data** | Value patterns are well-defined (phone numbers, dates, amounts) | Value patterns are irregular or contain natural language expressions |
| **Schema design** | Target cells can be pinpointed with `cell_id`, `bbox`, or `regex` | Fields are described by meaning via `description`, letting the LLM decide |
| **Infrastructure** | No GPU server needed (OCR only) | Requires an LLM server (e.g., vLLM) |
| **Speed** | Fast | Slower due to LLM inference |
| **Accuracy** | High accuracy when rules match | Strong on irregular data thanks to contextual understanding |

**When to use rule-based extraction:**

- Processing large volumes of identically formatted documents (application forms, inspection records, etc.)
- Extraction targets can be located by position or text pattern in advance
- Target cells can be directly specified with `cell_id` or `bbox`
- Value formats can be defined with regular expressions (invoice numbers, phone numbers, etc.)
- When an LLM server is unavailable or you want to minimize costs

**When to use LLM-based extraction:**

- Processing documents whose layout varies by issuer (business cards, receipts, etc.)
- Target text patterns are irregular and hard to identify with regex or keywords
- Extraction requires understanding the meaning or context of fields
- Rapid initial schema prototyping (iterate using only `description`)

!!! tip "Combined Workflow"
    Start with LLM-based extraction to understand a document's structure, then switch to rule-based extraction once the extraction rules are established. This optimizes both development efficiency and operational costs.

---

## Setup

### Basic Installation

```bash
pip install yomitoku
```

If you plan to use LLM-based extraction (`yomitoku_extract_with_llm`), install with the `extract` option:

```bash
pip install "yomitoku[extract]"
```

The `extract` option additionally installs the `openai` and `pyyaml` packages.

!!! note
    The `[extract]` option is not required if you only use rule-based extraction (`yomitoku_extract`).

### Setting Up vLLM Server (for LLM-based Extraction)

LLM-based extraction (`yomitoku_extract_with_llm`) requires an LLM server that provides an OpenAI-compatible API. We recommend [vLLM](https://docs.vllm.ai/).

#### 1. Install vLLM

```bash
pip install vllm
```

#### 2. Start the vLLM Server

```bash
vllm serve <model_name> \
    --host 0.0.0.0 \
    --port 8000 \
    --quantization awq \
    --dtype float16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.5
```

Example with Qwen3-4B-AWQ:

```bash
vllm serve Qwen/Qwen3-4B-AWQ \
    --host 0.0.0.0 \
    --port 8000 \
    --quantization awq \
    --dtype float16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.5
```

| Option | Description |
| :--- | :--- |
| `--quantization awq` | Enable AWQ quantization |
| `--dtype float16` | Set the data type to FP16. AWQ quantized models are not compatible with bf16, so `float16` must be specified explicitly |
| `--max-model-len` | Maximum sequence length. `8192` is sufficient for document extraction. Lower values reduce VRAM consumption |
| `--gpu-memory-utilization` | Upper limit of GPU memory that vLLM reserves (0.0–1.0). `0.5` limits allocation to 50% of VRAM. If the model does not fit within this limit, vLLM will fail to start. Set lower when sharing the GPU with yomitoku's OCR models |

Once the server is running, the OpenAI-compatible API will be available at `http://localhost:8000/v1`.

!!! note
    The vLLM server is not required for rule-based extraction (`yomitoku_extract`).

#### 3. Recommended Models

The following lightweight models with commercially permissive licenses are suitable for document data extraction.

| Model | Parameters | License | VRAM (approx.) | Features |
| :--- | :--- | :--- | :--- | :--- |
| [Qwen/Qwen3-4B-AWQ](https://huggingface.co/Qwen/Qwen3-4B-AWQ) | 4B (4-bit quantized) | Apache 2.0 | ~8GB | Low VRAM via AWQ quantization. Best balance of accuracy and efficiency. **Recommended** |
| [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | 4B | Apache 2.0 | ~15GB | Full-precision version. Highest accuracy but higher VRAM consumption |
| [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | 1.7B | Apache 2.0 | ~5–7GB | For minimal configurations. Low VRAM consumption but accuracy is limited |

!!! tip "Model Selection Guide"
    - **Recommended**: Qwen3-4B-AWQ. AWQ quantization delivers high accuracy with low VRAM, making it easy to share the GPU with yomitoku's OCR models
    - **Accuracy-focused**: Qwen3-4B (full-precision). For environments with ample VRAM
    - **Minimal setup**: Qwen3-1.7B. For environments with limited GPU VRAM

!!! warning "License Notice"
    All models listed above are licensed for commercial use. However, please review the license terms of each model before use. The Qwen3 series uses Apache 2.0 and Phi-4 uses the MIT license, both with no restrictions on commercial use.

Startup example (Qwen3-4B-AWQ):

```bash
vllm serve Qwen/Qwen3-4B-AWQ \
    --host 0.0.0.0 \
    --port 8000 \
    --quantization awq \
    --dtype float16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.5
```

---

## Schema Definition (YAML)

Fields to extract are defined in a YAML file. A schema consists of a `fields` list, where each field is either a KV (key-value) type or a table type.

!!! tip "Auto-generating Schemas"
    We recommend using generative AI such as ChatGPT or Claude to auto-generate schema definitions. Simply input the field names you want to extract (comma-separated), and the AI will generate YAML with appropriate `name`, `structure`, `type`, and `normalize` values. See [Schema Auto-generation Prompt](schema_generation_prompt.md) for the prompt template.

### Basic Structure

```yaml
fields:
  - name: <output_key>
    structure: kv
    description: <search_text>
    type: <value_type>
    normalize: <normalization_rule>

  - name: <output_key>
    structure: table
    columns:
      - name: <column_output_key>
        description: <column_header_search_text>
        type: <value_type>
        normalize: <normalization_rule>
```

---

### Common Field Parameters

| Parameter | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `name` | string | **Required** | - | Key name in the output JSON |
| `description` | string | - | `""` | Text used for searching (KV key text or column header text) |
| `cell_id` | string | - | `null` | Direct cell ID specification (e.g., `c12`) |
| `bbox` | list[int] | - | `null` | Bounding box position specification `[x1, y1, x2, y2]` |
| `regex` | string | - | `null` | Regex pattern for value extraction (scalar fields only) |
| `type` | string | - | `"string"` | Value type: `string`, `number`, `date` |
| `structure` | string | - | `"scalar"` | Data structure: `scalar`, `kv` (both KV format), or `table` |
| `normalize` | string | - | `null` | Normalization rule name |
| `columns` | list | - | `null` | Column definitions (for table-structure fields) |

---

### Specifying the Extraction Method

Each field (and each table column) supports four methods for identifying target cells.
When multiple methods are specified, they are evaluated in priority order: **`cell_id` > `bbox` > `description` > `regex`**.

#### 1. `description` - Text Search (Default)

Performs a partial match search against KV item key text or grid header text.

```yaml
- name: phone_number
  structure: kv
  description: Phone Number
  type: string
```

#### 2. `cell_id` - Cell ID Specification

Directly specify a cell ID assigned by the TableSemanticParser (e.g., `c12`, `c43`).
Cell IDs can be found in the table analysis result JSON or visualization images.

```yaml
- name: phone_number
  structure: kv
  cell_id: c43
  type: string
```

#### 3. `bbox` - Bounding Box Specification

Specify the cell's position as `[x1, y1, x2, y2]` and match by overlap (50% or more).

```yaml
- name: phone_number
  structure: kv
  bbox: [450, 120, 700, 160]
  type: string
```

#### 4. `regex` - Regular Expression Pattern (Scalar Fields Only)

Specify a regex pattern to extract the first matching string from cell and paragraph text. Useful when the key text is variable but the value format is known.

```yaml
# Extract by phone number pattern
- name: phone_number
  structure: kv
  regex: '\d{2,4}-\d{2,4}-\d{2,4}'
  type: string
  normalize: phone_jp

# Invoice number (T + 13 digits)
- name: invoice_number
  structure: kv
  regex: 'T\d{13}'
  type: string

# Email address
- name: email
  structure: kv
  regex: '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
  type: string
```

For `regex` matches, `value` contains the matched string and `raw_text` contains the full text of the source cell/paragraph.

!!! tip
    When both `description` and `cell_id` are specified, `cell_id` takes priority. If not found by `cell_id`, it falls back to `description`. `regex` is evaluated as a last resort when other methods fail.

---

### KV (Scalar) Fields

Used to extract key-value pairs from forms. Omit `structure` or set it to `scalar` or `kv`.

Explicitly specifying `kv` emphasizes in the LLM prompt that the field should be a scalar value (a single string), preventing the LLM from incorrectly outputting it as a table structure.

```yaml
fields:
  - name: company_name
    structure: kv
    description: Company Name
    type: string
    normalize: strip_spaces

  - name: total_amount
    structure: kv
    description: Total Amount
    type: number
    normalize: numeric
```

**Search order in rule-based extraction:**

1. If `cell_id` is specified, search directly by cell ID
2. If `bbox` is specified, search by position
3. If `description` is specified, search KV item key text
4. If `description` is specified, search cell text by partial match
5. If `description` is specified, search paragraph text by partial match
6. If `regex` is specified, search all cell, paragraph, and word text by regex
7. If none found, return an empty string

---

### Table Fields

Used to extract tabular data. Set `structure: table` and define columns with `columns`.

```yaml
fields:
  - name: order_items
    description: Order Details
    structure: table
    columns:
      - name: product
        description: Product Name
        type: string

      - name: quantity
        description: Quantity
        type: number
        normalize: numeric

      - name: price
        description: Amount
        type: number
        normalize: numeric
```

Column matching also supports `cell_id`, `bbox`, and `description`. These are matched against the grid's header cells.

```yaml
columns:
  - name: date
    cell_id: c8         # cell_id of the header cell
    type: date
    normalize: date_yyyymmdd

  - name: entrance_time
    description: Entry     # Match by header text
    type: string
    normalize: time_jp
```

---

### type (Value Type)

| Value | Description |
| :--- | :--- |
| `string` | String (default) |
| `number` | Number |
| `date` | Date |
| `alphanumeric` | Alphanumeric characters (half-width) |
| `hiragana` | Hiragana |
| `katakana` | Katakana |

`type` is primarily used as a hint in the prompt during LLM-based extraction. For rule-based extraction, the normalization rule (`normalize`) determines the actual value transformation.

---

### normalize (Normalization Rules)

Post-processing normalization is applied to extracted values. The following rules are available:

| Rule Name | Description | Input Example | Output Example |
| :--- | :--- | :--- | :--- |
| `strip_spaces` | Remove full-width and half-width spaces | `Tokyo Shibuya` | `TokyoShibuya` |
| `numeric` | Remove non-numeric characters (includes full-width to half-width conversion) | `1,234 yen` | `1234` |
| `phone_jp` | Format as Japanese phone number | `0312345678` | `03-1234-5678` |
| `postal_code_jp` | Format as Japanese postal code | `1234567` | `123-4567` |
| `date_jp` | Convert Japanese date to `YYYY-MM-DD` format | `令和6年3月15日`, `R6/3/15` | `2024-03-15` |
| `date_yyyymmdd` | Convert Japanese date to `YYYYMMDD` format | `令和6年3月15日`, `R6/3/15` | `20240315` |
| `time_jp` | Convert time to Japanese format | `14:30` | `14時30分` |
| `time_hms` | Convert time to `HH:MM:SS` format | `14時30分` | `14:30:00` |
| `alphanumeric` | Convert full-width to half-width, extract alphanumeric only | `ＡＢＣ１２３円` | `ABC123` |
| `hiragana` | Convert katakana to hiragana, extract hiragana only | `カタカナtest` | `かたかな` |
| `katakana` | Convert hiragana to katakana, extract katakana only | `ひらがなtest` | `ヒラガナ` |

If `normalize` is omitted, the extracted text is output as-is.

!!! note
    `date_jp` and `date_yyyymmdd` support Japanese era names in both kanji (令和, 平成, 昭和, 大正, 明治) and alphabetic abbreviations (R, H, S, T, M). Examples: `令和6年3月15日`, `R6/3/15`, `H30.1.1`, `S60-12-25`

---

### Schema Definition Example

The following example demonstrates how to define a schema using a facility usage application form.

![Facility usage application form example](assets/table.jpg){ width="600" }

This schema extracts the following information from the form:

- **KV fields**: Facility name, phone number, address, remarks
- **Table field**: Preferred usage dates (date, entry time, exit time)

```yaml
fields:
  # KV fields: search by key text using description
  - name: name
    structure: kv
    description: 施設名称       # "Facility Name"
    type: string
    normalize: strip_spaces

  - name: phone_number
    structure: kv
    description: 電話番号       # "Phone Number"
    type: string
    normalize: phone_jp

  - name: address
    structure: kv
    description: 住所           # "Address"
    type: string
    normalize: strip_spaces

  - name: details
    structure: kv
    description: 備考           # "Remarks"
    type: string
    normalize: strip_spaces

  # Table field: match columns by header text
  - name: usage_date
    description: 希望日         # "Preferred Date"
    structure: table
    columns:
      - name: date
        description: 日付       # "Date"
        type: date
        normalize: date_yyyymmdd

      - name: entrance_time
        description: 入室       # "Entry"
        type: string
        normalize: time_jp

      - name: leave_time
        description: 退室       # "Exit"
        type: string
        normalize: time_hms
```

**Key points:**

- For KV fields, specify the key text on the form (e.g., "施設名称", "電話番号") in `description`. The engine matches this against KV item keys and extracts the corresponding values
- For table fields, specify the header row text (e.g., "日付", "入室", "退室") in each column's `description`. The engine matches these against grid header cells and extracts data from each row
- Use `normalize` to specify post-extraction normalization. For example, `phone_jp` formats full-width digits into a hyphen-separated phone number, and `date_yyyymmdd` converts dates (including Japanese era formats) to `YYYYMMDD` format

---

## CLI Usage

### Rule-based Extraction

```bash
yomitoku_extract <input> -s <schema.yaml> [options]
```

| Option | Short | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `input` | - | string | **Required** | Path to input image, PDF, or directory |
| `--schema` | `-s` | string | **Required** | Path to schema definition file (YAML) |
| `--outdir` | `-o` | string | `results` | Output directory |
| `--device` | `-d` | string | `cuda` | Inference device (`cuda`, `cpu`, `mps`) |
| `--vis` | `-v` | - | `false` | Output visualization images |
| `--no-normalize` | - | - | `false` | Skip normalization |
| `--simple` | - | - | `false` | Output in simple format (without bbox metadata) |
| `--pages` | - | string | All pages | Pages to process (e.g., `1,3-5,10`) |
| `--dpi` | - | int | `200` | DPI for PDF loading |
| `--encoding` | - | string | `utf-8` | Output file encoding |

`input` accepts an image file, PDF file, or directory. When a directory is specified, it recursively searches for supported file formats (`jpg`, `jpeg`, `png`, `bmp`, `tiff`, `tif`, `pdf`) including subdirectories and processes them sequentially.

**Examples:**

```bash
# Extract from an image file
yomitoku_extract input.jpg -s schema.yaml -o results -v

# Process specific pages from a PDF
yomitoku_extract document.pdf -s schema.yaml --pages 1,3-5

# Batch process all files in a directory
yomitoku_extract ./documents/ -s schema.yaml -o results

# Run on CPU
yomitoku_extract input.jpg -s schema.yaml -d cpu
```

---

### LLM-based Extraction

```bash
yomitoku_extract_with_llm <input> -s <schema.yaml> -m <model_name> [options]
```

In addition to the rule-based extraction options, the following LLM-related options are available:

| Option | Short | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--model` | `-m` | string | **Required** | LLM model name (as specified on the vLLM server) |
| `--api-base` | - | string | `http://localhost:8000/v1` | LLM API base URL |
| `--api-key` | - | string | `""` | API key (if required) |
| `--temperature` | - | float | `0.0` | LLM generation temperature |
| `--max-tokens` | - | int | `4096` | LLM maximum token count |
| `--table-id` | - | string | `null` | Target a specific table only (e.g., `t0`) |
| `--simple` | - | - | `false` | Output in simple format (without bbox metadata) |
| `--prompt-only` | - | - | `false` | Print prompt and exit (for debugging) |

As with rule-based extraction, `input` accepts an image file, PDF file, or directory.

**Examples:**

```bash
# Extract using vLLM server
yomitoku_extract_with_llm input.jpg -s schema.yaml -m Qwen/Qwen3-4B-AWQ

# Batch process all files in a directory
yomitoku_extract_with_llm ./documents/ -s schema.yaml -m Qwen/Qwen3-4B-AWQ

# Specify API base URL and key
yomitoku_extract_with_llm input.jpg -s schema.yaml -m gpt-4o \
  --api-base https://api.openai.com/v1 \
  --api-key sk-xxxxx

# Check the prompt (debugging)
yomitoku_extract_with_llm input.jpg -s schema.yaml -m model_name --prompt-only
```

---

## Output Files

### Output Location

Output files are saved in the directory specified by `--outdir`. Filenames follow the format `<input_filename>_p<page_number>_extract.json`.

```
results/
  document_p1_extract.json
  document_p1_layout.jpg       # with --vis
  document_p1_ocr.jpg          # with --vis
  document_p1_extract_vis.jpg  # with --vis
```

### JSON Format

The output JSON has the following structure:

```json
{
  "fields": {
    "<field_name>": { ... }
  },
  "metadata": {
    "schema_version": "1.0"
  }
}
```

#### KV (Scalar) Field Output

```json
{
  "fields": {
    "phone_number": {
      "structure": "kv",
      "value": "03-1234-5678",
      "raw_text": "０３１２３４５６７８",
      "confidence": "high",
      "source": "cell_id",
      "cell_ids": ["c43"],
      "bboxes": [[450, 120, 700, 160]]
    }
  }
}
```

| Key | Description |
| :--- | :--- |
| `structure` | Always `"kv"` |
| `value` | Extracted value after normalization |
| `raw_text` | Original text before normalization |
| `confidence` | Confidence level (`high`, `medium`, `low`) |
| `source` | Extraction source (`cell_id`, `bbox`, `kv`, `paragraph`, `not_found`) |
| `cell_ids` | Cell IDs of the value |
| `bboxes` | Bounding boxes of the value |

#### Table Field Output

```json
{
  "fields": {
    "usage_date": {
      "structure": "table",
      "records": [
        {
          "date": {
            "value": "20240315",
            "cell_ids": ["c20"],
            "raw_text": "令和6年3月15日",
            "bboxes": [[100, 200, 300, 240]]
          },
          "entrance_time": {
            "value": "14時30分",
            "cell_ids": ["c21"],
            "raw_text": "14:30",
            "bboxes": [[310, 200, 450, 240]]
          }
        }
      ],
      "source": "grid"
    }
  }
}
```

| Key | Description |
| :--- | :--- |
| `structure` | Always `"table"` |
| `records` | Array of row data. Each row is a dictionary keyed by column name |
| `source` | Extraction source (`grid`, `not_found`) |

Each cell value includes `value` (after normalization), `raw_text` (before normalization), `cell_ids`, and `bboxes`.

### Simple Format

When the `--simple` option is specified, the output uses a simple `{name: value}` format without metadata such as bbox, cell_ids, confidence, and source.

#### KV (Scalar) Fields

```json
{
  "phone_number": "03-1234-5678",
  "company_name": "Test Corporation"
}
```

#### Table Fields

```json
{
  "usage_date": [
    {
      "date": "20240315",
      "entrance_time": "14時30分",
      "leave_time": "16:00:00"
    }
  ]
}
```

The simple format is convenient for downstream processing and system integration. Normalization is applied in the same way as the standard format.

### Visualization Images

When the `--vis` option is specified, the following images are output:

| Filename | Content |
| :--- | :--- |
| `*_layout.jpg` | Layout analysis result visualization |
| `*_ocr.jpg` | OCR result visualization |
| `*_extract_vis.jpg` | Highlighted positions of extracted fields (with field name labels) |

In the extraction visualization image, highlight intensity varies inversely with confidence level. Lower confidence fields are highlighted more intensely, making items that need review easy to spot (`low`: intense > `medium` > `high`: subtle).
