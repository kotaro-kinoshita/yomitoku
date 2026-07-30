# CLI Usage (Table Semantic Parser)

`yomitoku_table` コマンドは、Table Semantic Parser を使ってドキュメント全体を解析し、表の意味構造（Key-Value・グリッド）と段落をページ単位の構造化JSONとして出力します。

初回の実行時のみ、Hugging Face Hub からモデルの重みファイルをダウンロードします。

```bash
yomitoku_table ${path_data} -o results -v
```

| オプション名 | 説明 |
| :-- | :-- |
| `${path_data}` | 解析対象の画像・PDFファイル、またはそれらを含むディレクトリのパスを指定します。ディレクトリの場合はサブディレクトリも再帰的に処理します。 |
| `-o`, `--outdir` | 出力ディレクトリを指定します（なければ作成）。デフォルト: `results` |
| `-v`, `--vis` | 解析結果の可視化画像（`*_layout.jpg` / `*_ocr.jpg`）を出力します。 |
| `--vis_id` | `--vis` と併用すると、各セルの左上にセルIDを描画します。構造化JSONやテンプレートが参照するセルIDを画像上で確認できます。 |
| `-l`, `--lite` | 軽量モデルで実行します（CPU向け）。 |
| `-d`, `--device` | モデルを実行するデバイスを指定します（cuda \| cpu \| mps）。デフォルト: `cuda` |
| `--raw` | 正規化スキーマ（`TableSemanticParserSchema`）のJSONを出力します。 |
| `--simple` | 座標などのメタ情報を持たないテキストのみの構造化JSONを出力します。 |
| `--cell_name` | セル検出モデルを指定します。デフォルト: `rtdetrv2` |
| `--cell_cfg` | セル検出モデルの設定ファイル（YAML）のパスを指定します。 |
| `--lp_name` / `--lp_cfg` | レイアウト解析（テーブル検出）モデルの名前 / 設定ファイルを指定します。デフォルト: `rtdetrv2v2` |
| `--td_name` / `--td_cfg` | 文字検出モデルの名前 / 設定ファイルを指定します。デフォルト: `dbnetv2_1` |
| `--tr_name` / `--tr_cfg` | 文字認識モデルの名前 / 設定ファイルを指定します。デフォルト: `parseq-large-v4_1` |
| `--template` | テーブルテンプレートJSONを適用します（grid/kvの推論をスキップ）。 |
| `--grid_only` | グリッド領域のみを解析します（Key-Valueをスキップ）。 |
| `--kv_only` | Key-Valueのみを解析します（グリッドをスキップ）。 |
| `--pages` | 読み取り対象ページを指定します（例: `1,2,5-10`、1始まり）。デフォルト: 全ページ |
| `--dpi` | PDF読み込み時の解像度を指定します。デフォルト: `200` |
| `--encoding` | 出力ファイルの文字コードを指定します（`utf-8` \| `utf-8-sig` \| `shift-jis` \| `euc-jp` \| `cp932`）。 |

出力ファイルはページ単位で `{ファイル名}_p{ページ番号}.json` として保存されます。

## 出力形式

### デフォルト（構造化JSON）

`kv_items` / `grids` のセルIDをテキストに解決し、由来セルのIDと座標（`key_cells` / `value_cells`）を埋め込んだ構造化JSONを出力します。

- 同一の「キーセル」に複数の値が紐づく場合は、値を空間順（縦・横の並びを自動判定）に改行で結合し、`value_cells` に結合元のセルが順序どおり並びます。
- 結合の判定はキーの文字列ではなくセルIDで行うため、たまたま同じラベル文字列を持つ別のフィールドは結合されません。
- キーを持たない単独セル（`key` が空配列）も結合されず、個別のエントリのまま出力されます。

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
                    "key": ["利用情報", "施設名称"],
                    "value": "MLism株式会社",
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
                                    "key": ["日付"],
                                    "value": "2025年01月30日(月曜日)",
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
            "contents": "施設利用申込書"
        }
    ]
}
```

Python API からは `results.to_structured()` で同じ構造化ビューを取得できます。

### `--simple`（テキストのみ）

座標・セル参照を除いたテキストのみの形で出力します。`kv_items` はヘッダーの入れ子構造を保った階層的な辞書、グリッドの行は `{列ヘッダー: 値}` の辞書、段落は文字列の配列になります。

`kv_items` の階層化規則:

- 入れ子ヘッダー（親ヘッダー → 子ヘッダー）は入れ子の辞書になります
- 同じ階層に同名テキストの別ヘッダーが並ぶ場合（繰り返しブロック）は配列になります
- 親ヘッダーが値と子ヘッダーの両方を持つ場合、値は `_value` キーに入ります
- キーを持たない単独セルは予約キー `_unkeyed` の下に並びます

```bash
yomitoku_table ${path_data} -o results --simple
```

```json
{
    "tables": [
        {
            "id": "t0",
            "kv_items": {
                "利用情報": {
                    "施設名称": "MLism株式会社",
                    "利用目的": "セミナー"
                }
            },
            "grids": [
                {
                    "id": "g0",
                    "rows": [
                        {"日付": "2025年01月30日(月曜日)", "入室時刻": "10時00分", "退室時刻": "17時00分"}
                    ]
                }
            ]
        }
    ],
    "paragraphs": ["施設利用申込書", "下記のとおり、利用を申し込みます。"]
}
```

Python API からは `results.to_simple()` で取得できます。

### `--raw`（正規化スキーマ）

`TableSemanticParserSchema` をそのまま出力します。`cells`（セルIDをキーとする辞書）・`kv_items`（セルID参照）・`grids`・`words` を含むロスレスな形式で、テンプレートの往復や再解析に使用できます。詳細は[Table Semantic Parser](table_semantic_parser.ja.md)を参照してください。

```bash
yomitoku_table ${path_data} -o results --raw
```

## 軽量モードでの実行

`--lite` オプションを付与することで、軽量モデルを使用して推論します。CPU環境で高速に解析できますが、文字の認識精度が低下する可能性があります。

```bash
yomitoku_table ${path_data} --lite -d cpu
```

## モデル・Config の指定

各モジュールのモデル名と設定ファイル（YAML）を個別に指定できます。

```bash
yomitoku_table ${path_data} \
  --cell_name rtdetrv2 \
  --tr_name parseq-small \
  --td_cfg text_detector.yaml
```

| モジュール | 名前の指定 | Configの指定 | 選択肢 |
| :-- | :-- | :-- | :-- |
| セル検出 | `--cell_name` | `--cell_cfg` | `rtdetrv2`（正式版・入力960） |
| テーブル検出 | `--lp_name` | `--lp_cfg` | `rtdetrv2`, `rtdetrv2v2` |
| 文字検出 | `--td_name` | `--td_cfg` | `dbnet`, `dbnetv2`, `dbnetv2_1` |
| 文字認識 | `--tr_name` | `--tr_cfg` | `parseq`, `parseqv2`, `parseq-small`, `parseq-tiny`, `parseq-large-v4_1` |

## テンプレートの適用

`--template` にテンプレートJSONを指定すると、grid / kv の推論をスキップしてテンプレートの定義を適用します。テンプレートは `--raw` 出力の `save_template_json()` で作成できます。

```bash
yomitoku_table ${path_data} --template template.json
```

## グリッド / Key-Value のみを解析する

```bash
# グリッド (格子データ) のみ
yomitoku_table ${path_data} --grid_only

# Key-Value のみ
yomitoku_table ${path_data} --kv_only
```

## 読み取り対象ページを指定する

`--pages` で処理するページを指定します（1始まり、カンマ区切り・範囲指定可）。

```bash
yomitoku_table ${path_data} --pages 1,3-5
```

## 解析結果の可視化

`-v` を付与すると、ページごとに以下の画像を出力します。

- `*_layout.jpg` : テーブル・段落・セル役割（緑=ヘッダー、青=セル、マゼンタ=空セル）の可視化。確定した Key-Value のキー→値の連なりを**緑の矢印**、グリッドの構造を**青枠と矢印**で描画します。
- `*_ocr.jpg` : 文字検出・認識結果の可視化。

```bash
yomitoku_table ${path_data} -o results -v
```
