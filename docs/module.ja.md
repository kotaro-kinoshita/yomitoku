# Module Usage

Python ライブラリとして YomiToku を利用する場合の利用方法について説明します。

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
| `split_text_across_cells` | `bool` | 複数の表セルにまたがって検出されたテキスト行を、セル境界で分割してから文字認識するかどうかを指定します。 | デフォルトは `False` です。CLI では `--split_text_across_cells` を指定すると有効になります。 |
| `ignore_ruby` | `bool` | ふりがな（ルビ）テキストを出力から除外するかどうかを指定します。 | デフォルトは `False` です。 |
| `ruby_threshold` | `float` | ルビ判定の閾値を指定します。行高さの中央値に対する比率で、この閾値未満のひらがな・カタカナのみのテキストをルビとみなします。 | デフォルトは `0.5` です。`ignore_ruby=True` の場合のみ有効です。 |
| `enable_dewarp` | `bool` | 解析の前処理として幾何補正(歪み補正)を実行するかどうかを指定します。 | デフォルトは `False` です。詳細は[幾何補正(Dewarper)の利用](#dewarper)をご確認ください。 |
| `dewarp_coordinate` | `str` | 幾何補正を実行した場合の出力座標を指定します。(`dewarped` \| `photo`) | デフォルトは `"dewarped"`(補正後の画像の座標)です。 |

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

## 幾何補正(Dewarper)の利用 {: #dewarper}

撮影した文書画像の湾曲・カール・傾き・台形の歪みを補正します。

### Document Analyzer に組み込む

`enable_dewarp=True` を指定すると、解析の前処理として幾何補正を実行します。補正に失敗したと判定したページは補正せずに解析を続行し、`results.preprocess.dewarp.status` が `"rejected"` になります。

<!--codeinclude-->
[demo/simple_dewarp.py](../demo/simple_dewarp.py)
<!--/codeinclude-->

| 属性 | 説明 |
| :--- | :--- |
| `results.preprocess.dewarp` | 補正の結果(`status`, `reasons`, `coordinate`, `size` 等)。詳細は [CLI Usage](cli.ja.md) の幾何補正の節を参照してください。 |
| `analyzer.processed_img` | 解析に使用した画像。`dewarp_coordinate="dewarped"` で補正に成功した場合は補正後の画像です。 |
| `analyzer.preprocessor.dewarper.last_dewarped_img` | 直前の補正後の画像(`numpy.ndarray`, BGR)。補正に失敗した場合は `None` です。 |
| `analyzer.preprocessor.dewarper.visualize(img)` | 直前の補正の推定結果(紙の領域と Grid)を入力画像 `img` 上に描画した画像を返します。 |

`dewarp_coordinate="photo"` を指定すると、解析結果を元の画像の座標に逆投影して返します。各要素に元の画像上の曲がった多角形 `polygon` が付与され、`box` はその外接矩形になります。

### 補正後の画像の取得

補正後の画像は次のいずれかで取得できます。補正に失敗したと判定した場合(`status == "rejected"`)は補正後の画像はありません。

```python
import cv2

from yomitoku import DocumentAnalyzer, Dewarper
from yomitoku.data.functions import load_image

img = load_image("static/in/gallery4.jpg")[0]

# 1. Document Analyzer の前処理として補正した画像
analyzer = DocumentAnalyzer(device="cuda", enable_dewarp=True)
results, _, _ = analyzer(img)
if results.preprocess.dewarp.status == "ok":
    dewarped_img = analyzer.preprocessor.dewarper.last_dewarped_img  # dewarp_coordinate によらず補正後の画像
    # dewarp_coordinate="dewarped"(デフォルト)の場合は analyzer.processed_img も補正後の画像(解析に使用した画像)です
    cv2.imwrite("dewarped.jpg", dewarped_img)

# 2. Dewarper 単体: 第 2 戻り値が補正後の画像(rejected の場合は入力画像そのもの)
dewarper = Dewarper(device="cuda")
dewarp_results, dewarped_img = dewarper(img)
if dewarp_results.status == "ok":
    cv2.imwrite("dewarped.jpg", dewarped_img)
```

補正後の画像のサイズは `results.preprocess.dewarp.size`(`[幅, 高さ]`)にも記録されます。

### Dewarper 単体での利用

<!--codeinclude-->
[demo/simple_dewarper.py](../demo/simple_dewarper.py)
<!--/codeinclude-->

`Dewarper` は `(結果, 画像)` を返します。補正に成功した場合は補正後の画像、失敗と判定した場合は入力画像をそのまま返します。

| オプション名 | 型 | 説明 | 補足 |
| :--- | :--- | :--- | :--- |
| `device` | `str` | 処理に用いるデバイスを指定します。（指定値: `cuda` \| `cpu` \| `mps`） | デフォルトは `"cuda"` です。 |
| `path_cfg` | `str` | 補正の後処理や失敗判定の閾値を記述した YAML ファイルのパスを指定します。 | |

`DocumentAnalyzer` の `configs` では `{"preprocess": {"dewarper": {...}}}` に同じパラメータを指定します。

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
| `repetition_stop` | `bool` | （テキスト認識のみ）同じ文字・文字列が繰り返されるループを検出して打ち切るかどうかを指定します。未指定の場合はモデルconfigの値（既定: `True`）に従います。 |
| `rec_orientation_fallback` | `bool` | （テキスト認識のみ）信頼度が低い行を180度回転して再認識するかどうかを指定します。未指定の場合はモデルconfigの値（既定: `False`）に従います。 |
| `rec_orientation_fallback_thresh` | `float` | （テキスト認識のみ）認識方向フォールバックを実行する信頼度の閾値です。未指定の場合はモデルconfigの値（既定: `0.75`）に従います。 |

### サポートされるモデル名(`model_name`)

| モデルのタイプ | モデル名 |
| :--- | :--- |
| Text Recognizer (文字認識) | `"parseq-large-v4_1"`（デフォルト）, `"parseq-tiny-dynw-v4"`（軽量）, `"parseq"`, `"parseqv2"`, `"parseq-small"`, `"parseq-tiny"` |
| Text Detector (文字検出) | `"dbnetv2_1"`（デフォルト）, `"dbnet"`, `"dbnetv2"` |
| Layout Parser (レイアウト解析) | `"rtdetrv2"` |
| Table Structure Recognizer (表構造認識) | `"rtdetrv2"` |
| Dewarper (幾何補正) | `"uvdoc_v1"` |

`"parseq-tiny-dynw-v4"` は CLI の `--lite` オプションで使用される文字認識モデルです。
動的幅（dynamic width）モデルのため、高速に推論するには `dynamic_width=True` と
`batch_bucketing=True` を併用してください。

```python
from yomitoku import DocumentAnalyzer

configs = {
    "ocr": {
        "text_recognizer": {
            "model_name": "parseq-tiny-dynw-v4",
            "dynamic_width": True,
            "batch_bucketing": True,
        },
    },
}
DocumentAnalyzer(configs=configs)
```

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

引数・YAML・モデルの既定値の優先順位や、使い方ごとの指定場所は [設定の与え方](configuration.md#how-to-configure) を参照してください。

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
