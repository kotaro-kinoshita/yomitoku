# YomiToku

## 🌟 概要

YomiToku は日本語の文書画像解析に特化した AI ベースの文章画像解析エンジンです。画像内の文字の全文 OCR およびレイアウト解析機能を有しており、画像内の文字情報や図表を抽出、認識します。

- 🤖 独自に構築した日本語データセットで学習した 4 種類(文字位置の検知、文字列認識、レイアウト解析、表の構造認識)の AI の事前学習モデルを搭載しています。
- 🇯🇵 各モデルは日本語の文書画像に特化して学習されており、7000 文字を超える日本語文字の認識をサーポート、縦書きなど日本語特有のレイアウト構造の文書画像の解析も可能です
- 📈 レイアウト解析、表の構造解析機能により、文書画像のレイアウト構造を可能な限り維持した状態で、情報を抽出することが可能です。
- 📄 多様な出力形式をサポートし、html やマークダウン、json、csv のいずれかのフォーマットに変換し、出力可能です。

## 💡 インストールの方法

```
pip install --index-url https://test.pypi.org/simple/ yomitoku
```

### 依存ライブラリ

pdf ファイルの解析を行うためには、別途、[poppler](https://poppler.freedesktop.org/)のインストールが必要です。

**Mac**

```
brew install poppler
```

**Linux**

```
apt install poppler-utils -y
```

## 🚀 実行方法

```
yomitoku ${path_data} -f md -o results -v
```

- `${path_data}` 解析対象の画像が含まれたディレクトリか画像ファイルのパスを直接して指定してください。ディレクトリを対象とした場合はディレクトリのサブディレクトリ内の画像も含めて処理を実行します。
- `-f` 出力形式のファイルフォーマットを指定します。(json, csv, html, md をサポート)
- `-o` 出力先のディレクトリ名を指定します。存在しない場合は新規で作成されます。
- `-v` を指定すると解析結果を可視化した画像を出力します。
- `-d` モデルを実行するためのデバイスを指定します。gpu が利用できない場合は cpu で推論が実行されます。(デフォルト: cuda)

##　 📝 ドキュメント

パッケージの詳細は[ドキュメント](https://kotaro-kinoshita.github.io/yomitoku-dev/)を確認してください。

## LICENSE

YomiToku © 2024 by MLism inc. is licensed under CC BY-NC 4.0. To view a copy of this license, visit https://creativecommons.org/licenses/by-nc/4.0/
