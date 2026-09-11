# v5文字認識モデル

標準モデルは `parseq-middle-dynw-v5`、`-l` のモデルは `parseq-tiny-dynw-v5` です。
Studioも同じモデルconfigを使用します。
`--tr_name` を明示した場合は `-l` より優先されます。

v5では以下の設定がモデルconfigから適用されるため、可変幅推論のために
`-l` を指定する必要はありません。`--tr_cfg` のYAMLで上書きできます。

```yaml
data:
  dynamic_width: true
  batch_bucketing: true
  resize_policy: fit
  width_budget: 8000
  max_batch_size: 64
```

`fit` は縦横比を保ってキャンバス内に収め、小さい画像も拡大します。
`downscale` は縮小のみ行います。固定幅・可変幅のどちらにも同じpolicyを適用します。
旧モデルは `downscale`、v5は `fit` を使います。
Python APIの `dynamic_width` / `batch_bucketing` に明示した真偽値はconfigより優先します。
ONNXでは固定入力幅を使用します。

v5はcharset v3を使用し、NFKC一括正規化を行わず、専用の文字置換表で後処理します。

## HF登録

信頼できるローカル学習checkpointを指定します。最初に構造の厳密照合と
safetensors変換を行い、checkpointのSHA-256と推論設定をmanifestに保存します。

```bash
python scripts/upload_parseq_v5.py --size middle \
  --checkpoint /path/to/checkpoint.pth --save-dir /tmp/parseq-middle-v5
```

`--size tiny` でtinyを登録できます。`--push` を付けるとconfigに記載された
HFリポジトリへアップロードします。`--repo` で登録先、`--private` で新規作成時の
非公開設定を指定できます。認証はHFのログイン情報または `HF_TOKEN` を使用します。
