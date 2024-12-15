日本語版 | [English](README_EN.md)

<img src="static/logo/horizontal.png" width="800px">

![Python](https://img.shields.io/badge/Python-3.9|3.10|3.11|3.12-F9DC3E.svg?logo=python&logoColor=&style=flat)
![Pytorch](https://img.shields.io/badge/Pytorch-2.5-EE4C2C.svg?logo=Pytorch&style=fla)
![CUDA](https://img.shields.io/badge/CUDA->=11.8-76B900.svg?logo=NVIDIA&style=fla)
![OS](https://img.shields.io/badge/OS-Linux|Mac|Win-1793D1.svg?&style=fla)
[![Document](https://img.shields.io/badge/docs-live-brightgreen)](https://kotaro-kinoshita.github.io/yomitoku/)

## 🌟 概要

YomiToku は日本語に特化した AI 文章画像解析エンジン(Document AI)です。画像内の文字の全文 OCR およびレイアウト解析機能を有しており、画像内の文字情報や図表を認識、抽出、変換します。

- 🤖 日本語データセットで学習した 4 種類(文字位置の検知、文字列認識、レイアウト解析、表の構造認識)の AI モデルを搭載しています。4 種類のモデルはすべて独自に学習されたモデルで日本語文書に対して、高精度に推論可能です。
- 🇯🇵 各モデルは日本語の文書画像に特化して学習されており、7000 文字を超える日本語文字の認識をサーポート、縦書きなど日本語特有のレイアウト構造の文書画像の解析も可能です。（日本語以外にも英語の文書に対しても対応しています）。
- 📈 レイアウト解析、表の構造解析, 読み順推定機能により、文書画像のレイアウトの意味的構造を壊さずに情報を抽出することが可能です。
- 📄 多様な出力形式をサポートしています。html やマークダウン、json、csv のいずれかのフォーマットに変換可能です。また、文書内に含まれる図表、画像の抽出の出力も可能です。
- ⚡ GPU 環境で高速に動作し、効率的に文書の文字起こし解析が可能です。また、VRAM も 8GB 以内で動作し、ハイエンドな GPU を用意する必要はありません。

## 🖼️ デモ

[gallery.md](gallery.md)にも複数種類の画像の検証結果を掲載しています。

|                          入力画像                          |                       OCR の結果                        |
| :--------------------------------------------------------: | :-----------------------------------------------------: |
|        <img src="static/in/demo.jpg" width="400px">        | <img src="static/out/in_demo_p1_ocr.jpg" width="400px"> |
|                    レイアウト解析の結果                    |     エクスポート<br>(HTML で出力したものをスクショ)     |
| <img src="static/out/in_demo_p1_layout.jpg" width="400px"> |   <img src="static/out/demo_html.png" width="400px">    |

Markdown でエクスポートした結果は関してはリポジトリ内の[static/out/in_demo_p1.md](static/out/in_demo_p1.md)を参照

- `赤枠` : 図、画像等の位置
- `緑枠` : 表領域全体の位置
- `ピンク枠` : 表のセル構造(セル上の文字は [行番号, 列番号] (rowspan x colspan)を表します)
- `青枠` : 段落、テキストグループ領域
- `赤矢印` : 読み順推定の結果

画像の出典:[「令和 6 年版情報通信白書 3 章 2 節 AI の進化に伴い発展するテクノロジー」](https://www.soumu.go.jp/johotsusintokei/whitepaper/ja/r06/pdf/n1410000.pdf)：（総務省） を加工して作成

## 📣 リリース情報

- 2024 年 11 月 26 日 YomiToku v0.5.1 (beta) を公開

## 💡 インストールの方法

```
pip install yomitoku
```

- pytorch はご自身の CUDA のバージョンにあったものをインストールしてください。デフォルトでは CUDA12.4 以上に対応したものがインストールされます。
- pytorch は 2.5 以上のバージョンに対応しています。その関係で CUDA11.8 以上のバージョンが必要になります。対応できない場合は、リポジトリ内の Dockerfile を利用してください。

## 🚀 実行方法

```
yomitoku ${path_data} -f md -o results -v --figure　--lite
```

- `${path_data}` 解析対象の画像が含まれたディレクトリか画像ファイルのパスを直接して指定してください。ディレクトリを対象とした場合はディレクトリのサブディレクトリ内の画像も含めて処理を実行します。
- `-f`, `--format` 出力形式のファイルフォーマットを指定します。(json, csv, html, md をサポート)
- `-o`, `--outdir` 出力先のディレクトリ名を指定します。存在しない場合は新規で作成されます。
- `-v`, `--vis` を指定すると解析結果を可視化した画像を出力します。
- `-l`, `--lite` を指定すると軽量モデルで推論を実行します。通常より高速に推論できますが、若干、精度が低下する可能性があります。
- `-d`, `--device` モデルを実行するためのデバイスを指定します。gpu が利用できない場合は cpu で推論が実行されます。(デフォルト: cuda)
- `--ignore_line_break` 画像の改行位置を無視して、段落内の文章を連結して返します。（デフォルト：画像通りの改行位置位置で改行します。）
- `--figure_letter` 検出した図表に含まれる文字も出力ファイルにエクスポートします。
- `--figure` 検出した図、画像を出力ファイルにエクスポートします。(html と markdown のみ)

その他のオプションに関しては、ヘルプを参照

```
yomitoku --help
```

**NOTE**

- GPU での実行を推奨します。CPU を用いての推論向けに最適化されておらず、処理時間が長くなります。
- 活字のみ識別をサポートしております。手書き文字に関しては、読み取れる場合もありますが、公式にはサポートしておりません。
- Yomitoku は文書 OCR 向けに最適化されており、情景 OCR(看板など紙以外にプリントされた文字の読み取り)向けには最適化されていません。
- AI-OCR の識別精度を高めるために、入力画像の解像度が重要です。低解像度画像では識別精度が低下します。最低でも画像の短辺を 720px 以上の画像で推論することをお勧めします。

## 📝 ドキュメント

パッケージの詳細は[ドキュメント](https://kotaro-kinoshita.github.io/yomitoku/)を確認してください。

## LICENSE

本リポジトリ内に格納されているソースコードおよび本プロジェクトに関連する HuggingFaceHub 上のモデルの重みファイルのライセンスは CC BY-NC-SA 4.0 に従います。
非商用での個人利用、研究目的での利用はご自由にお使いください。
商用目的での利用に関しては、別途、商用ライセンスを提供しますので、https://www.mlism.com/ にお問い合わせください。

YomiToku © 2024 by Kotaro Kinoshita is licensed under CC BY-NC-SA 4.0. To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
