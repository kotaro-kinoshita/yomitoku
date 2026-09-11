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
  trailing_margin: 96
  width_budget: 8000
  max_batch_size: 64
```

`fit` は縦横比を保ってキャンバス内に収め、小さい画像も拡大します。
`downscale` は縮小のみ行います。固定幅・可変幅のどちらにも同じpolicyを適用します。
旧モデルは `downscale`、v5は `fit` を使います。
`trailing_margin` は可変幅画像の終端余白（px）です。v5のmiddle・tinyは96px、
未指定の旧configは64pxです。8px単位に切り上げ、キャンバス幅の上限800pxで制限します。
固定幅推論には影響しません。
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

## 終端マージンの評価（2026-09-11）

public PRのPython実装とmiddle v5の登録済み重み
（`checkpoint_best_middle_v5_20260911.pth`相当）を使用し、
diversity 20260911の508画像・47,021 GT枠で64pxと96pxを比較しました。
両条件ともnative PyTorch、TF32無効、追加リトライなしです。
GT枠固定の認識評価であり、検出器を含むE2E評価ではありません。
共通のNFKC・異体表・空白除去後に採点しています。

| 指標 | 64px | 96px |
| --- | ---: | ---: |
| CER | 6.2003% | 6.1745% |
| 枠の完全一致率 | 74.3817% | 74.4455% |
| 横書きCER（33,757枠） | 5.0353% | 5.0228% |
| 縦書きCER（13,264枠） | 8.1619% | 8.1136% |
| 挿入誤り（文字） | 3,301 | 3,253 |
| 削除誤り（文字） | 6,668 | 6,707 |
| 置換誤り（文字） | 20,025 | 19,909 |
| 過剰生成の目安（枠） | 47 | 47 |

編集距離は643枠で改善、543枠で悪化しました。222枠が正解になり、
192枠は正解から誤りになりました。全体の改善は小幅で、
繰り返しや縦書きの誤りが解消したわけではありません。
過剰生成の目安は「予測長がGTの2倍以上かつ8文字以上長い枠」であり、
繰り返しの確定判定ではありません。

今回の比較はPR実装で64px側も実行しており、前処理の異なる過去の
ローカル評価やStudio tiny ONNXの結果とは直接比較していません。
public/proのmiddle登録重みはSHA-256が一致しますが、全件評価に
使用した実装はpublicです。proではconfigから余白が反映されることを
テストしています。

- 重みSHA-256: `f3358d54b55731d732609b9806e1577f02e82be519b357c4ddbc6d7e6e280c1a`
- GT SHA-256: `3628d852e9c0f76284eaa0cc6443468f592765e715a3a5c37df5ed089fd0b2f5`

ローカルの `studio-evaluation/diversity-middle-v5-python-margin64-96/` に
評価スクリプト、設定・ソース差分manifest、予測全件、画像別指標、
変更枠CSV、HTMLレポートを保存しています。
