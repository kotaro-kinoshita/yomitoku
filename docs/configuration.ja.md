# Model Config

各モデルに対して、設定可能なパラメータについて説明します。

## 設定の与え方 {: #how-to-configure}

各モジュール（Text Detector / Text Recognizer / Layout Parser / Table Structure Recognizer）の設定は、次の 3 つの層で決まります。上の層ほど優先されます。

| 優先度 | 層 | 指定方法 |
| :---: | :--- | :--- |
| 高 | 引数 | CLI のオプション、Python API のコンストラクタ引数、`configs` 辞書に書いた値 |
| 中 | YAML | CLI の `--td_cfg` / `--tr_cfg` / `--lp_cfg` / `--tsr_cfg`、Python API の `path_cfg` |
| 低 | モデルの既定値 | `model_name` で選んだモデルの config に定義された値 |

- YAML には、変更したい項目だけを書けば十分です。書かなかった項目はモデルの既定値のままです。入れ子の項目（例: `data.batch_size`）も、変更する項目だけを書けます。
- 引数として指定できるのは、各モジュールのコンストラクタ引数として用意されている項目だけです（例: Text Recognizer の `repetition_stop`、`rec_orientation_fallback`）。閾値などそれ以外の項目は YAML で指定します。指定できる引数は [モジュールの利用](module.md) を参照してください。
- 同じ項目を YAML と引数の両方で指定した場合は、引数の値が使われます。

### 使い方ごとの指定場所

`configs` 辞書のキーは使い方によって異なります。各キーの中には、そのモジュールの引数（`path_cfg` を含む）を辞書で書きます。

| 使い方 | Text Detector | Text Recognizer | Layout Parser | Table Structure Recognizer |
| :--- | :--- | :--- | :--- | :--- |
| CLI（`yomitoku`） | `--td_cfg` | `--tr_cfg` | `--lp_cfg` | `--tsr_cfg` |
| `DocumentAnalyzer(configs=...)` | `["ocr"]["text_detector"]` | `["ocr"]["text_recognizer"]` | `["layout_analyzer"]["layout_parser"]` | `["layout_analyzer"]["table_structure_recognizer"]` |
| `OCR(configs=...)` | `["text_detector"]` | `["text_recognizer"]` | — | — |
| `LayoutAnalyzer(configs=...)` | — | — | `["layout_parser"]` | `["table_structure_recognizer"]` |
| `TableSemanticParser(configs=...)` | `["text_detector"]` | `["text_recognizer"]` | `["table_detector"]` | — |
| モジュール単体 | `TextDetector(...)` | `TextRecognizer(...)` | `LayoutParser(...)` | `TableStructureRecognizer(...)` |

`TableSemanticParser` のセル検出モデル（Table Cell Parser）の設定は `configs["table_cell_parser"]` に書きます。

### 例: 文字認識の設定を変える

`tr.yaml` に閾値などを書きます。

```yaml
# tr.yaml
rep_min_run_p1: 24     # 1 文字の繰り返しを 24 回まで許容する
data:
  batch_size: 32       # data の他の項目は既定値のまま
```

CLI では `--tr_cfg` で YAML を渡し、ON/OFF はオプションで指定します。

```bash
yomitoku ${path_data} --tr_cfg tr.yaml --disable-repetition-stop
```

Python API では、`path_cfg` と引数を同じ辞書に書きます。

```python
from yomitoku import OCR, DocumentAnalyzer

# DocumentAnalyzer: "ocr" の下に書く
analyzer = DocumentAnalyzer(
    configs={
        "ocr": {
            "text_recognizer": {
                "path_cfg": "tr.yaml",
                "repetition_stop": False,
                "rec_orientation_fallback": True,
            },
        },
    },
)

# OCR: configs の直下に書く
ocr = OCR(
    configs={
        "text_recognizer": {
            "path_cfg": "tr.yaml",
            "repetition_stop": False,
        },
    },
)
```

### YAML を書くときの注意

- 存在しない項目を書くと、読み込み時にエラーになります。項目名のタイプミスに気付けるよう、エラーメッセージを確認してください。

    ```
    omegaconf.errors.ConfigKeyError: Key 'repetition_stopp' not in 'TextRecognizer...Config'
    ```

- YAML の値は、`model_name` で選んだモデルの既定値に上書きされます。モデルを切り替えても、同じ YAML をそのまま使えます。
- モデルのネットワーク構造に関する項目（`encoder` / `decoder` など）は、学習済みの重みと一致しなくなるため変更しないでください。

### 適用された設定を確認する

各モジュールの `log_config()` で、引数と YAML を反映した最終的な設定をログに出力できます。`save_config(path)` で YAML ファイルに保存することもできます。

```python
analyzer.text_recognizer.log_config()   # DocumentAnalyzer の場合
ocr.recognizer.log_config()             # OCR の場合
```

保存されるファイルには、環境に依存する値（文字セットファイルの絶対パスなど）も含まれます。設定の確認用として使い、`path_cfg` に渡す YAML には変更したい項目だけを書いてください。

## Text Detector

文字の領域を検出するモデルです。入力画像のサイズと、検出結果の後処理を調整できます。

| 項目 | 型 | 既定値 | 説明 |
| :--- | :--- | :--- | :--- |
| `data.shortest_size` | `int` | `1280` | 画像の短辺がこの値より小さい場合、短辺がこの値になるように拡大します。小さい文字の検出漏れが多い場合に大きくします。 |
| `data.limit_size` | `int` | `1600` | 画像の長辺がこの値より大きい場合、長辺がこの値以下になるように縮小します。大きくすると細かい文字を検出しやすくなる一方、処理時間とメモリ使用量が増えます。 |
| `post_process.thresh` | `float` | モデルにより異なる（下表） | 予測ヒートマップを二値化する閾値です。この値を超えるピクセルを文字領域として扱います。下げると薄い文字も拾いやすくなり、上げるとノイズが減ります。 |
| `post_process.box_thresh` | `float` | モデルにより異なる（下表） | 領域内の平均スコアがこの値を下回る領域を除外します。 |
| `post_process.min_size` | `int` | `2` | 領域の短辺がこの値を下回る場合に除外します。 |
| `post_process.max_candidates` | `int` | `1500` | 1 ページで処理する領域数の上限です。文字数が非常に多いページで検出漏れがある場合に大きくします。 |
| `post_process.unclip_ratio` | `float` | モデルにより異なる（下表） | 検出領域を外側に広げる度合いです。大きいほど余白を持たせた検出になり、小さいほどタイトな検出になります。 |
| `visualize.heatmap` | `bool` | `false` | 可視化画像に予測ヒートマップを重ねて描画するかどうかです。 |
| `visualize.color` | `[B, G, R]` | `[0, 255, 0]` | 可視化画像に描画する検出枠の色です。 |

モデルごとの既定値:

| モデル | `post_process.thresh` | `post_process.box_thresh` | `post_process.unclip_ratio` |
| :--- | :--- | :--- | :--- |
| `dbnet` | `0.15` | `0.5` | `7.0` |
| `dbnetv2` | `0.2` | `0.5` | `5.0` |
| `dbnetv2_1` | `0.3` | `0.4` | `3.5` |

`backbone.*` / `decoder.*` はネットワーク構造の設定で、学習済みの重みと一致させる必要があるため変更できません。

## Text Recognizer

切り出した文字領域の文字列を認識するモデルです。「引数」列に ✔ がある項目は、コンストラクタ引数や `configs` 辞書でも指定でき、その値が YAML より優先されます。

### 推論とバッチ処理

| 項目 | 型 | 既定値 | 引数 | 説明 |
| :--- | :--- | :--- | :---: | :--- |
| `data.batch_size` | `int` | モデルにより異なる（下表） |  | 1 回の推論でまとめて処理する文字領域の数です（`width_budget` を使う場合を除く）。大きくすると GPU では高速になる一方、メモリ使用量が増えます。 |
| `data.dynamic_width` | `bool` | モデルにより異なる（下表） | ✔ | 切り出し画像を固定幅までパディングせず、内容の幅に合わせて推論します。対応モデルでのみ有効にしてください。ONNX 推論では常に無効です。 |
| `data.batch_bucketing` | `bool` | モデルにより異なる（下表） | ✔ | 幅の近い切り出し画像を同じバッチにまとめます。`dynamic_width` と併用したときに効果があります。 |
| `data.width_budget` | `int` | モデルにより異なる（下表） |  | `dynamic_width` 有効時、バッチ内の「画像数 × 最大幅」がこの値を超えないようにバッチを区切ります。GPU メモリが不足する場合は小さくします。 |
| `data.max_batch_size` | `int` | モデルにより異なる（下表） |  | `width_budget` 使用時の 1 バッチあたりの画像数の上限です。 |
| `data.num_workers` | `int` | `8` |  | 切り出し画像の前処理に使うスレッド数です。 |
| `decode_ar` | `int` | `1` |  | `1` で自己回帰デコード、`0` で非自己回帰デコードを行います。`0` は高速ですが、認識結果が変わり精度が下がる場合があります。 |
| `refine_iters` | `int` | `1` |  | デコード後に結果を見直す反復回数です。`0` にすると処理が軽くなります。 |

モデルごとの既定値（`—` はそのモデルの config に項目がなく、指定できないことを表します）:

| モデル | `data.batch_size` | `data.dynamic_width` | `data.batch_bucketing` | `data.width_budget` | `data.max_batch_size` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `parseq` | `128` | `false` | `false` | — | — |
| `parseqv2` | `128` | `false` | `false` | — | — |
| `parseq-small` | `128` | `false` | `false` | — | — |
| `parseq-tiny` | `128` | `false` | `false` | — | — |
| `parseq-large-v4_1` | `128` | `false` | `false` | — | — |
| `parseq-tiny-dynw-v4` | `10` | `true` | `true` | `8000` | `64` |
| `parseq-tiny-dynw-v5` | `10` | `true` | `true` | `8000` | `64` |
| `parseq-middle-dynw-v5` | `10` | `true` | `true` | `8000` | `64` |

### 繰り返し抑制

| 項目 | 型 | 既定値 | 引数 | 説明 |
| :--- | :--- | :--- | :---: | :--- |
| `repetition_stop` | `bool` | `true` | ✔ | 同じ文字・文字列が繰り返されるループを検出した時点でデコードを打ち切り、繰り返しを 1 回分に縮約します。PyTorch 推論でのみ動作し、ONNX 推論では適用されません。CLI では `--disable-repetition-stop` で無効化できます。 |
| `rep_period_max` | `int` | `8` |  | 繰り返しとみなす単位の最大長（トークン数）です。 |
| `rep_min_run_p1` | `int` | `16` |  | 1 文字の繰り返しを抑制するのに必要な最小連続回数です。リーダー（`……`）など長い連続を残したい場合は大きくします。 |
| `rep_min_repeats` | `int` | `3` |  | 2 文字以上の単位の繰り返しを抑制するのに必要な最小繰り返し回数です。 |

### 認識方向フォールバック

| 項目 | 型 | 既定値 | 引数 | 説明 |
| :--- | :--- | :--- | :---: | :--- |
| `rec_orientation_fallback` | `bool` | `false` | ✔ | 信頼度が低い行を 180 度回転して再認識し、信頼度が高い方の結果を採用します。CLI では `--enable-rec-orientation-fallback` で有効化できます。 |
| `rec_orientation_fallback_thresh` | `float` | `0.75` | ✔ | フォールバックを実行する信頼度の閾値です。CLI では `--rec-orientation-fallback-thresh` で指定できます。 |

### 可視化

| 項目 | 型 | 既定値 | 引数 | 説明 |
| :--- | :--- | :--- | :---: | :--- |
| `visualize.font` | `str` | モデルにより異なる |  | 可視化画像に認識結果を描画するフォントファイル（.ttf）のパスです。 |
| `visualize.color` | `[B, G, R]` | `[0, 0, 255]` |  | 認識結果の文字色です（Text Recognizer 単体と Table Semantic Parser の可視化で使用）。 |
| `visualize.font_size` | `int` | `18` |  | 認識結果の文字の大きさです（Text Recognizer 単体と Table Semantic Parser の可視化で使用）。 |

### モデルに合わせた値を使う項目

次の項目は変更できますが、モデルの学習時の条件に合わせてあります。変更すると精度が下がるため、通常はモデルの値をそのまま使ってください。

- `data.resize_policy`: 切り出し画像の拡大縮小の方法（`fit` は小さい画像も拡大、`downscale` は縮小のみ）
- `data.trailing_margin`: 動的幅推論で行末に加える余白
- `nfkc_normalize`: 認識結果に NFKC 正規化をかけるかどうか
- `char_replace_table`: 認識結果の文字を置き換える対応表のパス

`data.trailing_margin` / `nfkc_normalize` / `char_replace_table` は、config にこれらの項目があるモデルでのみ指定できます。

### 変更できない項目

次の項目は学習済みの重みと一致させる必要があるため変更できません。変更すると、モデルの読み込み時または推論時にエラーになります。

- `max_label_length`（認識できる最大文字数）
- `data.img_size`（入力画像サイズ）
- `charset` / `num_tokens`（文字セット）
- `encoder.*` / `decoder.*`（ネットワーク構造）
- `hf_hub_repo`（重みの取得先）

このほか、`TextRecognizer` の引数でのみ指定できる項目として、`num_parallel_batches`（並列に推論するバッチ数）と `source_downscale`（切り出し前に高解像度の画像を縮小するかどうか）があります。

## Layout Parser / Table Structure Recognizer / Table Cell Parser

レイアウト・表に関するモデルで調整できるのは、予測スコアの閾値 `thresh_score` です。値を上げると確信度の高い領域だけが残り、下げると検出漏れが減る一方で誤検出が増えます。

| モデル | 用途 | `thresh_score` の既定値 |
| :--- | :--- | :--- |
| Layout Parser | `DocumentAnalyzer` / `LayoutAnalyzer` のレイアウト解析 | `0.5` |
| Table Structure Recognizer | `DocumentAnalyzer` / `LayoutAnalyzer` の表構造認識 | `0.4` |
| Table Cell Parser | `TableSemanticParser` のセル検出（`configs["table_cell_parser"]`） | `0.5` |

`TableSemanticParser` の表検出（`configs["table_detector"]`）には Layout Parser が使われるため、同じ `thresh_score` で調整できます。

`data.img_size`、`category`、ネットワーク構造（`PResNet.*` / `HybridEncoder.*` / `RTDETRTransformerv2.*`）はモデル固有の値で、変更できません。`data.img_size` を変更すると推論時にエラーになります。
