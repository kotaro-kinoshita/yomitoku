# Yomitoku Extractor

Yomitoku Extractorは、帳票画像やPDFからスキーマ定義に従って構造化データを抽出するツールです。
YomiTokuのテーブル解析エンジン（TableSemanticParser）でOCR・レイアウト解析を行い、その結果からYAMLで定義したスキーマに基づいてフィールドの値を抽出します。

抽出方式は以下の2種類から選択できます。

| コマンド | 抽出方式 | 特徴 |
| :--- | :--- | :--- |
| `yomitoku_extract` | ルールベース | LLM不要。KV検索・グリッド照合で高速に抽出 |
| `yomitoku_extract_with_llm` | LLMベース | vLLMなどのLLMサーバーを利用してより柔軟に抽出 |

### 使い分けの指針

2つの抽出方式は帳票の種類や抽出対象の特性に応じて使い分けます。

| 観点 | `yomitoku_extract`（ルールベース） | `yomitoku_extract_with_llm`（LLMベース） |
| :--- | :--- | :--- |
| **適した帳票** | 定型帳票（申請書、報告書、伝票など） | 非定型帳票（名刺、レシート、請求書など） |
| **レイアウト** | レイアウトが固定で毎回同じ位置に同じ項目がある | レイアウトが発行元によって異なる |
| **抽出対象** | 抽出する値のパターンが明確（電話番号、日付、金額など） | 値の出現パターンが不定、または自然言語的な表現を含む |
| **スキーマ設計** | `cell_id`・`bbox`・`regex` で対象を正確に特定できる | `description` でフィールドの意味を記述し、LLMに判断を委ねる |
| **実行環境** | GPUサーバー不要（OCRのみ） | vLLM等のLLMサーバーが必要 |
| **処理速度** | 高速 | LLM推論のため低速 |
| **精度** | ルールに合致すれば高精度 | 文脈を理解できるため非定型データに強い |

**ルールベース抽出が適するケース:**

- 同一フォーマットの帳票を大量に処理する場合（申請書、点検記録など）
- 抽出対象の位置やテキストパターンが事前に特定できる場合
- `cell_id` や `bbox` で対象セルを直接指定できる場合
- 正規表現で値のフォーマットが定義できる場合（インボイス番号、電話番号など）
- LLMサーバーを用意できない、またはコストを抑えたい場合

**LLMベース抽出が適するケース:**

- 名刺やレシートなど、発行元ごとにレイアウトが異なる帳票を処理する場合
- 抽出対象のテキストパターンが不規則で、正規表現やキーワードでは特定が難しい場合
- フィールドの意味や文脈を理解して抽出する必要がある場合
- 初回のスキーマ設計を迅速に行いたい場合（`description` のみで試行錯誤できる）

!!! tip "組み合わせ運用"
    まずLLMベース抽出で帳票の構造を把握し、抽出ルールが確定したらルールベース抽出に切り替えることで、開発効率と運用コストの両方を最適化できます。

---

## 環境構築

### 基本インストール

```bash
pip install yomitoku
```

LLMベース抽出（`yomitoku_extract_with_llm`）を利用する場合は、`extract` オプションを付けてインストールします。

```bash
pip install "yomitoku[extract]"
```

`extract` オプションにより、`openai` および `pyyaml` パッケージが追加インストールされます。

!!! note
    ルールベース抽出（`yomitoku_extract`）のみ使用する場合、`[extract]` オプションは不要です。

### vLLMサーバーの構築（LLMベース抽出を使う場合）

LLMベースの抽出（`yomitoku_extract_with_llm`）を利用する場合、OpenAI互換APIを提供するLLMサーバーが必要です。[vLLM](https://docs.vllm.ai/) を推奨します。

#### 1. vLLMのインストール

```bash
pip install vllm
```

#### 2. vLLMサーバーの起動

```bash
vllm serve <model_name> \
    --host 0.0.0.0 \
    --port 8000 \
    --quantization awq \
    --dtype float16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.5
```

例: Qwen3-4B-AWQを使用する場合

```bash
vllm serve Qwen/Qwen3-4B-AWQ \
    --host 0.0.0.0 \
    --port 8000 \
    --quantization awq \
    --dtype float16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.5
```

| オプション | 説明 |
| :--- | :--- |
| `--quantization awq` | AWQ量子化を有効にする |
| `--dtype float16` | データ型を FP16 に指定する。AWQ量子化モデルは bf16 非対応のため `float16` を明示する |
| `--max-model-len` | 最大シーケンス長。帳票抽出では `8192` で十分。小さくするとVRAM消費を削減できる |
| `--gpu-memory-utilization` | vLLMが確保するGPUメモリの上限割合（0.0〜1.0）。`0.5` でVRAMの50%までに制限する。収まらない場合は起動エラーになる。yomitokuのOCRモデルと同一GPUで動かす場合は低めに設定する |

サーバーが起動したら `http://localhost:8000/v1` でOpenAI互換APIが利用可能になります。

!!! note
    ルールベース抽出（`yomitoku_extract`）の場合、vLLMサーバーは不要です。

#### 3. 推奨モデル

帳票データ抽出に適した、商用利用可能なライセンスの軽量モデルを以下に示します。

| モデル | パラメータ数 | ライセンス | VRAM目安 | 特徴 |
| :--- | :--- | :--- | :--- | :--- |
| [Qwen/Qwen3-4B-AWQ](https://huggingface.co/Qwen/Qwen3-4B-AWQ) | 4B (4bit量子化) | Apache 2.0 | 約8GB | AWQ量子化により省VRAM。精度と速度のバランスに優れる。**推奨** |
| [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | 4B | Apache 2.0 | 約15GB | 非量子化版。精度は最も高いがVRAM消費が大きい |
| [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | 1.7B | Apache 2.0 | 約5〜7GB | 最小構成向け。VRAM消費は小さいが精度は限定的 |

!!! tip "モデル選定の指針"
    - **推奨**: Qwen3-4B-AWQ。AWQ量子化により少ないVRAMで高い精度を実現。yomitokuのOCRモデルとGPUを共有しやすい
    - **精度重視**: Qwen3-4B（非量子化）。VRAMに余裕がある環境向け
    - **最小構成**: Qwen3-1.7B。GPU VRAMが限られる環境向け

!!! warning "ライセンスに関する注意"
    上記モデルはいずれも商用利用が許可されたライセンスです。ただし、利用前に各モデルのライセンス条項を必ず確認してください。特に Qwen3 シリーズは Apache 2.0、Phi-4 は MIT ライセンスで、いずれも商用利用に制限はありません。

起動例（Qwen3-4B-AWQ）:

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

## スキーマ定義（YAML）

抽出対象のフィールドをYAMLファイルで定義します。スキーマは `fields` リストで構成され、各フィールドにはKV（キー・バリュー）形式とテーブル形式の2種類があります。

!!! tip "スキーマの自動生成"
    スキーマ定義は ChatGPT や Claude などの生成AIを使って自動生成することを推奨します。抽出したい項目名をカンマ区切りで入力するだけで、適切な `name`・`structure`・`type`・`normalize` を含むYAMLを生成できます。プロンプトのテンプレートは [スキーマ自動生成プロンプト](schema_generation_prompt.md) を参照してください。

### 基本構造

```yaml
fields:
  - name: <出力キー名>
    structure: kv
    description: <検索文字列>
    type: <値の型>
    normalize: <正規化ルール>

  - name: <出力キー名>
    structure: table
    columns:
      - name: <列の出力キー名>
        description: <列ヘッダーの検索文字列>
        type: <値の型>
        normalize: <正規化ルール>
```

---

### フィールド共通パラメータ

| パラメータ | 型 | 必須 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- | :--- |
| `name` | string | **必須** | - | 出力JSONのキー名 |
| `description` | string | - | `""` | 検索に使用するテキスト（KVのキー文字列やカラムヘッダーのテキスト） |
| `cell_id` | string | - | `null` | セルIDによる直接指定（例: `c12`） |
| `bbox` | list[int] | - | `null` | バウンディングボックスによる位置指定 `[x1, y1, x2, y2]` |
| `regex` | string | - | `null` | 正規表現パターンによる値の抽出（スカラーフィールドのみ） |
| `type` | string | - | `"string"` | 値の型。`string`, `number`, `date` |
| `structure` | string | - | `"scalar"` | データ構造。`scalar`、`kv`（いずれもKV形式）、または `table` |
| `normalize` | string | - | `null` | 正規化ルール名 |
| `columns` | list | - | `null` | テーブル構造の場合の列定義 |

---

### 抽出方法の指定

各フィールド（およびテーブルの各列）には、対象セルの特定方法を4種類から指定できます。
複数指定した場合は **`cell_id` > `bbox` > `description` > `regex`** の優先順で評価されます。

#### 1. `description` - テキスト検索（デフォルト）

KVアイテムのキー文字列やグリッドのヘッダー文字列に対して部分一致で検索します。

```yaml
- name: phone_number
  structure: kv
  description: 電話番号
  type: string
```

#### 2. `cell_id` - セルID指定

TableSemanticParserが割り当てたセルID（例: `c12`, `c43`）を直接指定します。
テーブル解析結果のJSONやビジュアライゼーション画像からセルIDを確認できます。

```yaml
- name: phone_number
  structure: kv
  cell_id: c43
  type: string
```

#### 3. `bbox` - バウンディングボックス指定

セルの位置情報 `[x1, y1, x2, y2]` を指定し、オーバーラップ判定（50%以上）で照合します。

```yaml
- name: phone_number
  structure: kv
  bbox: [450, 120, 700, 160]
  type: string
```

#### 4. `regex` - 正規表現パターン（スカラーフィールドのみ）

正規表現パターンを指定し、セルやパラグラフのテキストから最初にマッチした文字列を抽出します。キー文字列が不定でも値のフォーマットが既知の場合に有効です。

```yaml
# 電話番号パターンで抽出
- name: phone_number
  structure: kv
  regex: '\d{2,4}-\d{2,4}-\d{2,4}'
  type: string
  normalize: phone_jp

# インボイス番号（T+13桁）
- name: invoice_number
  structure: kv
  regex: 'T\d{13}'
  type: string

# メールアドレス
- name: email
  structure: kv
  regex: '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
  type: string
```

`regex` のマッチ結果は `value` にマッチ文字列、`raw_text` にマッチ元のセル/パラグラフ全文が格納されます。

!!! tip
    `description` と `cell_id` を併記した場合、`cell_id` が優先されます。`cell_id` で見つからなかった場合に `description` にフォールバックします。`regex` は他の方法で見つからなかった場合の最終手段として評価されます。

---

### KV（スカラー）フィールド

帳票内のキーと値のペアを抽出する場合に使用します。`structure` を省略するか、`scalar` または `kv` を指定します。

`kv` を明示的に指定すると、LLMベース抽出時にプロンプトで当該フィールドがスカラー値（単一の文字列）であることが強調され、LLMがテーブル構造として誤出力することを防ぎます。

```yaml
fields:
  - name: company_name
    structure: kv
    description: 会社名
    type: string
    normalize: strip_spaces

  - name: total_amount
    structure: kv
    description: 合計金額
    type: number
    normalize: numeric
```

**ルールベース抽出での検索順序:**

1. `cell_id` が指定されている場合、セルIDで直接検索
2. `bbox` が指定されている場合、位置情報で検索
3. `description` が指定されている場合、KVアイテムのキー文字列で検索
4. `description` が指定されている場合、セルのテキストで部分一致検索
5. `description` が指定されている場合、パラグラフのテキストで部分一致検索
6. `regex` が指定されている場合、全セル・パラグラフ・ワードのテキストを正規表現で検索
7. いずれも見つからなければ空文字を返却

---

### テーブルフィールド

表形式のデータを抽出する場合に使用します。`structure: table` を指定し、`columns` で列を定義します。

```yaml
fields:
  - name: order_items
    description: 注文明細
    structure: table
    columns:
      - name: product
        description: 商品名
        type: string

      - name: quantity
        description: 数量
        type: number
        normalize: numeric

      - name: price
        description: 金額
        type: number
        normalize: numeric
```

列の照合でも `cell_id`、`bbox`、`description` が使用できます。これらはグリッドのヘッダーセルに対して照合されます。

```yaml
columns:
  - name: date
    cell_id: c8         # ヘッダーセルのcell_id
    type: date
    normalize: date_yyyymmdd

  - name: entrance_time
    description: 入室     # ヘッダーのテキストで照合
    type: string
    normalize: time_jp
```

---

### type（値の型）

| 値 | 説明 |
| :--- | :--- |
| `string` | 文字列（デフォルト） |
| `number` | 数値 |
| `date` | 日付 |
| `alphanumeric` | 半角英数字 |
| `hiragana` | ひらがな |
| `katakana` | カタカナ |

`type` は主にLLMベース抽出時にプロンプトへのヒントとして使用されます。ルールベース抽出では、正規化ルール（`normalize`）の指定が実際の値変換に影響します。

---

### normalize（正規化ルール）

抽出した値に対して後処理として正規化を適用します。以下のルールが利用可能です。

| ルール名 | 説明 | 入力例 | 出力例 |
| :--- | :--- | :--- | :--- |
| `strip_spaces` | 全角・半角空白を除去 | `東京都　渋谷区` | `東京都渋谷区` |
| `numeric` | 数値以外を除去（全角→半角変換含む） | `１，２３４円` | `1234` |
| `phone_jp` | 日本の電話番号形式に整形 | `０３１２３４５６７８` | `03-1234-5678` |
| `postal_code_jp` | 日本の郵便番号形式に整形 | `１２３４５６７` | `123-4567` |
| `date_jp` | 日本語日付を `YYYY-MM-DD` 形式に変換 | `令和6年3月15日`, `R6/3/15` | `2024-03-15` |
| `date_yyyymmdd` | 日本語日付を `YYYYMMDD` 形式に変換 | `令和6年3月15日`, `R6/3/15` | `20240315` |
| `time_jp` | 時刻を `X時XX分` 形式に変換 | `14:30` | `14時30分` |
| `time_hms` | 時刻を `HH:MM:SS` 形式に変換 | `14時30分` | `14:30:00` |
| `alphanumeric` | 全角→半角変換し、英数字のみを抽出 | `ＡＢＣ１２３円` | `ABC123` |
| `hiragana` | カタカナ→ひらがな変換し、ひらがなのみを抽出 | `カタカナtest` | `かたかな` |
| `katakana` | ひらがな→カタカナ変換し、カタカナのみを抽出 | `ひらがなtest` | `ヒラガナ` |

`normalize` を省略した場合、抽出されたテキストがそのまま出力されます。

!!! note
    `date_jp` および `date_yyyymmdd` は和暦に対応しています。漢字表記（令和・平成・昭和・大正・明治）とアルファベット略称（R・H・S・T・M）の両方を認識します。例: `令和6年3月15日`, `R6/3/15`, `H30.1.1`, `S60-12-25`

---

### スキーマ定義例

以下の施設利用申請書を例に、スキーマの定義方法を説明します。

![施設利用申請書の例](assets/table.jpg){ width="600" }

この帳票から以下の情報を抽出するスキーマを定義します。

- **KVフィールド**: 施設名称、電話番号、住所、備考
- **テーブルフィールド**: 利用希望日（日付・入室時刻・退室時刻）

```yaml
fields:
  # KVフィールド: descriptionでキー文字列を指定して検索
  - name: name
    structure: kv
    description: 施設名称
    type: string
    normalize: strip_spaces

  - name: phone_number
    structure: kv
    description: 電話番号
    type: string
    normalize: phone_jp

  - name: address
    structure: kv
    description: 住所
    type: string
    normalize: strip_spaces

  - name: details
    structure: kv
    description: 備考
    type: string
    normalize: strip_spaces

  # テーブルフィールド: 列ヘッダーのテキストで列を照合
  - name: usage_date
    description: 希望日
    structure: table
    columns:
      - name: date
        description: 日付
        type: date
        normalize: date_yyyymmdd

      - name: entrance_time
        description: 入室
        type: string
        normalize: time_jp

      - name: leave_time
        description: 退室
        type: string
        normalize: time_hms
```

**スキーマのポイント:**

- KVフィールドでは `description` に帳票上のキー文字列（「施設名称」「電話番号」など）を指定します。エンジンがKVアイテムのキーと照合し、対応する値を抽出します
- テーブルフィールドでは `columns` の各 `description` にヘッダー行のテキスト（「日付」「入室」「退室」）を指定します。グリッドのヘッダーセルと照合し、各行のデータを抽出します
- `normalize` で抽出値の正規化を指定します。例えば `phone_jp` は全角数字をハイフン区切りの電話番号形式に、`date_yyyymmdd` は和暦を含む日付を `YYYYMMDD` 形式に変換します

---

## CLI実行方法

### ルールベース抽出

```bash
yomitoku_extract <input> -s <schema.yaml> [options]
```

| オプション | 短縮形 | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- | :--- |
| `input` | - | string | **必須** | 入力画像、PDFのパス、またはディレクトリ |
| `--schema` | `-s` | string | **必須** | スキーマ定義ファイル（YAML）のパス |
| `--outdir` | `-o` | string | `results` | 出力ディレクトリ |
| `--device` | `-d` | string | `cuda` | 推論デバイス（`cuda`, `cpu`, `mps`） |
| `--vis` | `-v` | - | `false` | 可視化画像を出力 |
| `--no-normalize` | - | - | `false` | 正規化処理をスキップ |
| `--simple` | - | - | `false` | シンプルフォーマットで出力（bbox等のメタデータなし） |
| `--pages` | - | string | 全ページ | 処理対象ページ（例: `1,3-5,10`） |
| `--dpi` | - | int | `200` | PDF読み取り時のDPI |
| `--encoding` | - | string | `utf-8` | 出力ファイルの文字コード |

`input` には画像ファイル、PDFファイル、またはディレクトリを指定できます。ディレクトリを指定した場合、サブディレクトリを含めて再帰的に対応形式（`jpg`, `jpeg`, `png`, `bmp`, `tiff`, `tif`, `pdf`）のファイルを検索し、順番に処理します。

**実行例:**

```bash
# 画像ファイルから抽出
yomitoku_extract input.jpg -s schema.yaml -o results -v

# PDFの特定ページのみ処理
yomitoku_extract document.pdf -s schema.yaml --pages 1,3-5

# ディレクトリ内の全ファイルを一括処理
yomitoku_extract ./documents/ -s schema.yaml -o results

# CPU環境で実行
yomitoku_extract input.jpg -s schema.yaml -d cpu
```

---

### LLMベース抽出

```bash
yomitoku_extract_with_llm <input> -s <schema.yaml> -m <model_name> [options]
```

ルールベース抽出のオプションに加え、以下のLLM関連オプションが利用可能です。

| オプション | 短縮形 | 型 | デフォルト | 説明 |
| :--- | :--- | :--- | :--- | :--- |
| `--model` | `-m` | string | **必須** | LLMモデル名（vLLMサーバーで指定したモデル名） |
| `--api-base` | - | string | `http://localhost:8000/v1` | LLM APIのベースURL |
| `--api-key` | - | string | `""` | APIキー（必要な場合） |
| `--temperature` | - | float | `0.0` | LLMの生成温度 |
| `--max-tokens` | - | int | `4096` | LLMの最大トークン数 |
| `--table-id` | - | string | `null` | 特定テーブルのみ対象にする（例: `t0`） |
| `--simple` | - | - | `false` | シンプルフォーマットで出力（bbox等のメタデータなし） |
| `--prompt-only` | - | - | `false` | プロンプトを出力して終了（デバッグ用） |

`input` はルールベース抽出と同様に、画像ファイル、PDFファイル、またはディレクトリを指定できます。

**実行例:**

```bash
# vLLMサーバーを使用した抽出
yomitoku_extract_with_llm input.jpg -s schema.yaml -m Qwen/Qwen3-4B-AWQ

# ディレクトリ内の全ファイルを一括処理
yomitoku_extract_with_llm ./documents/ -s schema.yaml -m Qwen/Qwen3-4B-AWQ

# APIベースURLとキーを指定
yomitoku_extract_with_llm input.jpg -s schema.yaml -m gpt-4o \
  --api-base https://api.openai.com/v1 \
  --api-key sk-xxxxx

# プロンプトの確認（デバッグ）
yomitoku_extract_with_llm input.jpg -s schema.yaml -m model_name --prompt-only
```

---

## 出力ファイル

### 出力先

出力ファイルは `--outdir` で指定したディレクトリに保存されます。ファイル名は `<入力ファイル名>_p<ページ番号>_extract.json` の形式です。

```
results/
  document_p1_extract.json
  document_p1_layout.jpg       # --vis 指定時
  document_p1_ocr.jpg          # --vis 指定時
  document_p1_extract_vis.jpg  # --vis 指定時
```

### JSONフォーマット

出力JSONは以下の構造を持ちます。

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

#### KV（スカラー）フィールドの出力

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

| キー | 説明 |
| :--- | :--- |
| `structure` | `"kv"` 固定 |
| `value` | 正規化後の抽出値 |
| `raw_text` | 正規化前の元テキスト |
| `confidence` | 信頼度（`high`, `medium`, `low`） |
| `source` | 抽出元（`cell_id`, `bbox`, `kv`, `paragraph`, `not_found`） |
| `cell_ids` | 値のセルID |
| `bboxes` | 値のバウンディングボックス |

#### テーブルフィールドの出力

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

| キー | 説明 |
| :--- | :--- |
| `structure` | `"table"` 固定 |
| `records` | 行データの配列。各行は列名をキーとする辞書 |
| `source` | 抽出元（`grid`, `not_found`） |

各セルの値には `value`（正規化後）、`raw_text`（正規化前）、`cell_ids`、`bboxes` が含まれます。

### シンプルフォーマット

`--simple` オプションを指定すると、bbox・cell_ids・confidence・source等のメタデータを含まないシンプルな `{name: value}` 形式で出力されます。

#### KV（スカラー）フィールド

```json
{
  "phone_number": "03-1234-5678",
  "company_name": "株式会社テスト"
}
```

#### テーブルフィールド

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

シンプルフォーマットは後続の処理やシステム連携で利用しやすい形式です。正規化は通常フォーマットと同様に適用されます。

### 可視化画像

`--vis` オプションを指定すると以下の画像が出力されます。

| ファイル名 | 内容 |
| :--- | :--- |
| `*_layout.jpg` | レイアウト解析結果の可視化 |
| `*_ocr.jpg` | OCR結果の可視化 |
| `*_extract_vis.jpg` | 抽出されたフィールドの位置をハイライト表示（フィールド名ラベル付き） |

抽出可視化画像では、信頼度に応じてハイライトの濃さが変化します。信頼度が低いほどハイライトが濃く表示され、確認が必要な箇所を直感的に把握できます（`low`：濃い > `medium` > `high`：薄い）。
