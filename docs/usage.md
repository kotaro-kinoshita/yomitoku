# Usage
## OCR
```bash
uv run python examples/simple_ocr.py  --vis --image $PATH_IMAGE --config $PATH_CONFIG
```

- デフォルトのconfigファイルはconfigs/ocr.yamlにあります
- config内の重みファイルのパスWEIGHTSを任意のパスに書き換える
- --visオプションを使用すると、出力先に予測結果の可視化画像を出力します。