# Usage

## CLI からの利用

初回の実行時のみ, HuggingFaseHub からモデルの重みファイルをダウンロードします。

```
yomitoku ${path_data} -f md -o results -v
```

- `${path_data}` 解析対象の画像が含まれたディレクトリか画像ファイルのパスを直接して指定してください。かディレクトリを対象とした場合はディレクトリのサブディレクトリ内の画像も含めて処理を実行します。入力をサポートしているファイル形式は jpeg, png, bmp, tiff, pdf です。
- `-f` 出力形式のファイルフォーマットを指定します。(json, csv, html, md をサポート)
- `-l` 指定すると軽量モデルで推論を実行します。CPU でも高速に推論可能です。
- `-o` 出力先のディレクトリ名を指定します。存在しない場合は新規で作成されます。
- `-v` を指定すると解析結果を可視化した画像を出力します。
- `-d` モデルを実行するためのデバイスを指定します。gpu が利用できない場合は cpu で推論が実行されます。(デフォルト: cuda)

### Note:

- CPU を用いての推論向けに最適化されておらず、処理時間が長くなりますので、GPU を用いてでも実行を推奨します。
- 活字のみの識別をサポートしております。手書き文字に関しては、読み取れる場合もありますが、公式にはサポートしておりません。
- OCR は文書 OCR と情景 OCR(看板など紙以外にプリントされた文字)に大別されますが、Yomitoku は文書 OCR 向けに最適化されています。
- AI-OCR の識別精度を高めるために、入力画像の解像度が重要です。低解像度画像では識別精度が低下します。画像の短辺を 1000px 以上の画像で推論することをお勧めします。

## Python のコード内での利用

### Document Analyzer の利用

Document Analyzer は OCR およびレイアウト解析を実行し、それらの結果を統合した解析結果を返却します。段落、表の構造解析、抽出、図表の検知など様々なユースケースにご利用いただけます。

<!--codeinclude-->

[demo/simple_document_analysis.py](../demo/simple_document_analysis.py)

<!--/codeinclude-->

- `visualize` を True にすると各処理結果を可視化した結果を第２、第 3 戻り値に OCR、レアウト解析の処理結果をそれぞれ格納し、返却します。False にした場合は None を返却します。描画処理のための計算が増加しますので、デバック用途でない場合は、False を推奨します。
- `device` には処理に用いる計算機を指定します。Default は"cuda". GPU が利用できない場合は、自動で CPU モードに切り替えて処理を実行します。
- `configs`を活用すると、パイプラインの処理のより詳細のパラメータを設定できます。

`DocumentAnalyzer` の処理結果のエクスポートは以下に対応しています。

- `to_json()`: JSON 形式(\*.json)
- `to_html()`: HTML 形式(\*.html)
- `to_csv()`: カンマ区切り CSV 形式(\*.csv)
- `to_markdown()`: マークダウン形式(\*.md)

### AI-OCR のみの利用

AI-OCR では、テキスト検知と検知したテキストに対して、認識処理を実行し、画像内の文字の位置と読み取り結果を返却します。

<!--codeinclude-->

[demo/simple_ocr.py](../demo/simple_ocr.py)

<!--/codeinclude-->

- `visualize` を True にすると各処理結果を可視化した結果を第２、第 3 戻り値に OCR、レアウト解析の処理結果をそれぞれ格納し、返却します。False にした場合は None を返却します。描画処理のための計算が増加しますので、デバック用途でない場合は、False を推奨します。
- `device` には処理に用いる計算機を指定します。Default は"cuda". GPU が利用できない場合は、自動で CPU モードに切り替えて処理を実行します。
- `configs`を活用すると、パイプラインの処理のより詳細のパラメータを設定できます。

`OCR`の処理結果のエクスポートは JSON 系形式(`to_json()`)のみサポートしています。

### Layout Analyzer のみの利用

LayoutAnalyzer では、テキスト検知と検知したテキストに対して、段落、図表の検知および表の構造解析処理 AI を実行し、文書内のレイアウト構造を解析します。

<!--codeinclude-->

[demo/simple_layout.py](../demo/simple_layout.py)

<!--/codeinclude-->

- `visualize` を True にすると各処理結果を可視化した結果を第２、第 3 戻り値に OCR、レアウト解析の処理結果をそれぞれ格納し、返却します。False にした場合は None を返却します。描画処理のための計算が増加しますので、デバック用途でない場合は、False を推奨します。
- `device` には処理に用いる計算機を指定します。Default は"cuda". GPU が利用できない場合は、自動で CPU モードに切り替えて処理を実行します。
- `configs`を活用すると、パイプラインの処理のより詳細のパラメータを設定できます。

`LayoutAnalyzer`の処理結果のエクスポートは JSON 系形式(`to_json()`)のみサポートしています。

## パイプラインの詳細設定

Config を与えることで、より細かい振る舞いを調整できます。

### Config の記述方法

config は辞書形式で与えます。config を与えることでモジュールごとに異なる計算機で処理を実行したり、詳細のパラーメタの設定が可能です。例えば以下のような config を与えると、OCR 処理は GPU で実行し、レイアウト解析機能は CPU で実行します。

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

### yaml ファイルでのパラメータの定義

Config に yaml ファイルのパスを与えることで、推論時の細部のパラメータの調整が可能です。yaml ファイルの例はリポジトリ内の`configs`ディレクトリ内にあります。モデルのネットワークのパラメータは変更できませんが、後処理のパラメータや入力画像のサイズなどは一部変更が可能です。

たとえば、以下のように`Text Detector`の後処理の閾値を yaml を定義し、config にパスを設定することができます。config ファイルはすべてのパラメータを記載する必要はなく、変更が必要なパラメータのみの記載が可能です。

`text_detector.yaml`の記述

```yaml
post_process:
  thresh: 0.1
  unclip_ratio: 2.5
```

yaml ファイルのパスを config に格納する

<!--codeinclude-->

[demo/setting_document_anaysis.py](../demo/setting_document_anaysis.py)

<!--/codeinclude-->

## インターネットに接続できない環境での利用

Yomitoku は初回の実行時に HuggingFaceHub からモデルを自動でダウンロードします。その際にインターネット環境が必要ですが、事前に手動でダウンロードすることでインターネットに接続できない環境でも実行することが可能です。

1. [Git Large File Storage](https://docs.github.com/ja/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)をインストール

2. 事前にインターネットに接続できる環境でモデルリポジトリをダウンロードします。クローンしたリポジトリはご自身のツールで動作環境にコピーしてください。

以下は huggingfacehub からモデルリポジトリをダウンロードするコマンド

```sh
git clone https://huggingface.co/KotaroKinoshita/yomitoku-table-structure-recognizer-rtdtrv2-open-beta

git clone https://huggingface.co/KotaroKinoshita/yomitoku-layout-parser-rtdtrv2-open-beta

git clone https://huggingface.co/KotaroKinoshita/yomitoku-text-detector-dbnet-open-beta

git clone https://huggingface.co/KotaroKinoshita/yomitoku-text-recognizer-parseq-open-beta
```

3. yomitoku のリポジトリの直下にモデルリポジトリを配置し、yaml ファイルの`hf_hub_repo`でローカルのモデルレポジトリを参照します。以下は `text_detector.yaml` の例です。同様に他のモジュールに対しても yaml ファイルを定義します。

```yaml
hf_hub_repo: yomitoku-text-detector-dbnet-open-beta
```

4. yaml ファイルのパスを config に格納する

<!--codeinclude-->

[demo/setting_document_anaysis.py](../demo/setting_document_anaysis.py)

<!--/codeinclude-->
