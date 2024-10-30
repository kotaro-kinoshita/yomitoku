# YomiToku-dev
YomiTokuの開発用リポジトリ

# Setup
## リポジトリからのインストール
- 仮想環境のセットアップ
```
uv sync --dev
```

## TestPyPIからのインストール
```
pip install --index-url https://test.pypi.org/simple/ yomitoku
```


- 重みファイルのダウンロードし、実行フォルダに配置
https://drive.google.com/drive/folders/1xMClVygBcK8pJnxKn8Cd7vIJoTBw4Nut?usp=sharing


# Demo
```
uv run python examples/simple_ocr.py  --vis --image ${PATH_IMAGE} --config ${PATH_CONFIG}
```
- デフォルトのconfigファイルは`configs/ocr.yaml`にあります
- config内の重みファイルのパス`WEIGHTS`を任意のパスに書き換える
- --visオプションは出力先に予測結果の可視化画像を出力する