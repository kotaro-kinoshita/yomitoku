# Module Usage

## Document Analyzer の利用

Document Analyzer は OCR およびレイアウト解析を実行し、それらの結果を統合した解析結果を返却します。段落、表の構造解析、抽出、図表の検知など様々なユースケースにご利用いただけます。

以下の 4 つのモデルがモジュール内で使われます。

- Text Recognizer (文字認識)
- Text Detector (文字検出)
- Layout Parser (レイアウト解析)
- Table Structure Recognizer (表構造認識)

<!--codeinclude-->
[demo/simple_document_analysis.py](../demo/simple_document_analysis.py)
<!--/codeinclude-->

| オプション名 | 型 | 説明 | 補足 |
| :--- | :--- | :--- | :--- |
| `visualize` | `bool` | 処理結果の可視化を行うかどうかを指定します。 | デバッグ用途でない場合は `False` を推奨します。`True` の場合、第 2 戻り値に OCR 結果、第 3 戻り値にレイアウト解析結果を返却します。`False` の場合は `None` を返却します。 |
| `device` | `str` | 処理に用いるデバイスを指定します。 | デフォルトは `"cuda"` です。GPU が利用できない場合は、自動で `"cpu"` に切り替わります。 |
| `configs` | `dict` | モジュールの処理のより詳細なパラメータを設定するために利用します。 | 詳細は[モデルの詳細設定](#model-config)をご確認ください。 |

`DocumentAnalyzer` の処理結果のエクスポートは以下に対応しています。

| メソッド | 出力形式 |
| :--- | :--- |
| `to_json()` | JSON 形式 (\*.json) |
| `to_html()` | HTML 形式 (\*.html) |
| `to_csv()` | カンマ区切り CSV 形式 (\*.csv) |
| `to_markdown()` | マークダウン形式 (\*.md) |

## AI-OCR のみの利用

AI-OCR では、テキスト検知と検知したテキストに対して、認識処理を実行し、画像内の文字の位置と読み取り結果を返却します。

以下の 2 つのモデルがモジュール内で使われます。

- Text Recognizer (文字認識)
- Text Detector (文字検出)

<!--codeinclude-->
[demo/simple_ocr.py](../demo/simple_ocr.py)
<!--/codeinclude-->

| オプション名 | 型 | 説明 | 補足 |
| :--- | :--- | :--- | :--- |
| `visualize` | `bool` | 処理結果の可視化を行うかどうかを指定します。 | デバッグ用途でない場合は `False` を推奨します。`True` の場合、第 2 戻り値に OCR 結果を返却します。`False` の場合は `None` を返却します。 |
| `device` | `str` | 処理に用いるデバイスを指定します。（指定値: `cuda` \| `cpu` \| `mps`） | デフォルトは `"cuda"` です。GPU が利用できない場合は、自動で `"cpu"` に切り替わります。 |
| `configs` | `dict` | モジュールの処理のより詳細なパラメータを設定するために利用します。 | 詳細は[モデルの詳細設定](#model-config)をご確認ください。 |

`OCR`の処理結果のエクスポートは JSON 系形式(`to_json()`)のみサポートしています。

## Layout Analyzer のみの利用

Layout Analyzer では、テキスト検知と検知したテキストに対して、段落、図表の検知および表の構造解析処理 AI を実行し、文書内のレイアウト構造を解析します。

以下の 2 つのモデルがモジュール内で使われます

- Layout Parser (レイアウト解析)
- Table Structure Recognizer (表構造認識)

<!--codeinclude-->
[demo/simple_layout.py](../demo/simple_layout.py)
<!--/codeinclude-->

| オプション名 | 型 | 説明 | 補足 |
| :--- | :--- | :--- | :--- |
| `visualize` | `bool` | 処理結果の可視化を行うかどうかを指定します。 | デバッグ用途でない場合は `False` を推奨します。`True` の場合、第 2 戻り値にレイアウト解析結果を返却します。`False` の場合は `None` を返却します。 |
| `device` | `str` | 処理に用いるデバイスを指定します。（指定値: `cuda` \| `cpu` \| `mps`） | デフォルトは `"cuda"` です。GPU が利用できない場合は、自動で `"cpu"` に切り替わります。 |
| `configs` | `dict` | モジュールの処理のより詳細なパラメータを設定するために利用します。 | 詳細は[モデルの詳細設定](#model-config)をご確認ください。 |

`LayoutAnalyzer`の処理結果のエクスポートは JSON 系形式(`to_json()`)のみサポートしています。

## モデルの詳細設定 {: #model-config}

Config を与えることで、より細かい振る舞いを調整できます。モデルに対して、以下のパラメータを設定可能です。

| オプション名 | 型 | 説明 |
| :--- | :--- | :--- |
| `model_name` | `str` | 使用するモデルの名前を指定します。 |
| `path_cfg` | `str` | ハイパーパラメータが記述された config ファイルのパスを入力します。 |
| `device` | `str` | 推論に使用するデバイスを指定します。（指定値: `cuda` \| `cpu` \| `mps`） |
| `visualize` | `bool` | 可視化処理を実施するかどうかを指定します。 |
| `from_pretrained` | `bool` | Pretrained モデル（学習済みモデル）を使用するかどうかを指定します。 |
| `infer_onnx` | `bool` | PyTorch の代わりに ONNX Runtime を使用して推論するかどうかを指定します。 |

### サポートされるモデル名(`model_name`)

| モデルのタイプ | モデル名 |
| :--- | :--- |
| Text Recognizer (文字認識) | `"parseq"`, `"parseq-small"`, `"parseq-tiny"` |
| Text Detector (文字検出) | `"dbnet"` |
| Layout Parser (レイアウト解析) | `"rtdetrv2"` |
| Table Structure Recognizer (表構造認識) | `"rtdetrv2"` |

### Config の記述方法

Config は辞書形式で与えます。Config を与えることでモデルごとに異なるデバイスで処理を実行したり、詳細のパラメータの設定が可能です。

例えば以下のような Config を与えると、OCR 処理は GPU で実行し、レイアウト解析機能は CPU で実行します。

```python
from yomitoku import DocumentAnalyzer

if __name__ == "__main__":
    configs = {
        "ocr": {
            "text_detector": {
                "device": "cuda",
            },
            "text_recognizer": {
                "device": "cuda",
            },
        },
        "layout_analyzer": {
            "layout_parser": {
                "device": "cpu",
            },
            "table_structure_recognizer": {
                "device": "cpu",
            },
        },
    }

    DocumentAnalyzer(configs=configs)
```

## YAML ファイルでのパラメータの定義

Config に YAML ファイルのパスを与えることで、推論時の細部のパラメータの調整が可能です。YAML ファイルの例はリポジトリ内の`configs`ディレクトリ内にあります。
モデルのネットワークのパラメータは変更できませんが、後処理のパラメータや入力画像のサイズなどは一部変更が可能です。変更可能なパラメータについては[Model Config](configuration.ja.md)をご確認ください。

例えば、以下のように Text Detector の後処理の閾値を YAML ファイルに定義し、Config にその YAML ファイルのパスを設定することができます。Config ファイルはすべてのパラメータを記載する必要はなく、変更が必要なパラメータのみの記載が可能です。

`text_detector.yaml`の記述

```yaml
post_process:
  thresh: 0.1
  unclip_ratio: 2.5
```

以下のように YAML ファイルのパスを Config に格納可能です。

<!--codeinclude-->
[demo/setting_document_anaysis.py](../demo/setting_document_anaysis.py)
<!--/codeinclude-->
