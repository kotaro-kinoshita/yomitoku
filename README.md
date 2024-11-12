# YomiToku-dev

YomiToku の開発用リポジトリ

# Setup

## リポジトリからのインストール

- 仮想環境のセットアップ

```
uv sync --dev
```

## TestPyPI からのインストール

```
pip install --index-url https://test.pypi.org/simple/ yomitoku
```

# Demo

```
uv run python examples/simple_ocr.py  --vis --image ${PATH_IMAGE}
```

- --vis オプションは出力先に予測結果の可視化画像を出力する
- --image 入力画像のパス
