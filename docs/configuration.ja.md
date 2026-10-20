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
| `TableSemanticParser(configs=...)` | `["text_detector"]` | `["text_recognizer"]` | — | — |
| モジュール単体 | `TextDetector(...)` | `TextRecognizer(...)` | `LayoutParser(...)` | `TableStructureRecognizer(...)` |

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

### 入力画像サイズに関する設定

```yaml
data:
  # 画像の短辺ピクセル数が設定した数値を下回る場合にここで設定した画像のピクセル数に以上になるように画像を拡大します。
  shortest_size: int 

  # 画像の長辺ピクセル数が設定した数値を上回る場合にここで設定した画像のピクセル数以下になるように画像を縮小します。
  limit_size: int 
```

### 後処理

```yaml
post_process:
  # 検出した領域の辺の大きさが設定した数値を下回る場合に領域を除去します。
  min_size: int 

  # モデルの予測スコアに対する閾値で、予測スコアが設定した閾値を下回るピクセルを背景領域として扱います。
  thresh: float 

  # 領域内の予測の平均スコアに対する閾値で、閾値を下回る領域を除外します。
  box_thresh: float 

  # 検出可能なテキスト領域数の上限
  max_candidates: int 

  # テキスト領域のマージン領域の大きさを設定するためのパラメータ。大きいほど、テキスト領域のマージンを大きくし、余白を持たせた検出が可能になり、小さいほどタイトな検出になります。
  unclip_ratio: int 
```

### 可視化設定

```yaml
visualize:
  # 検出領域のバウンディングボックスの色の設定
  color: [B, G, R] 

  # モデルの予測ヒートマップを可視化、描画するか
  heatmap: boolean 
```

## Text Recognizer

### 文字列長

```yaml
# 予測可能な最大文字列長
max_label_length: int
```

### 入力画像

```yaml
data:
   # バッチ処理に用いる画像数
  batch_size: int
```

### 認識方向フォールバック

```yaml
# 信頼度が低い場合にROI画像を180度回転して再認識を行うかどうか
rec_orientation_fallback: bool

# フォールバックを実行する信頼度の閾値
rec_orientation_fallback_thresh: float
```

デフォルト値は全モデル共通で、`rec_orientation_fallback: false`、`rec_orientation_fallback_thresh: 0.75` です。CLI の `--enable-rec-orientation-fallback` / `--rec-orientation-fallback-thresh`、または Python API の `TextRecognizer(rec_orientation_fallback=..., rec_orientation_fallback_thresh=...)` で指定した値はモデルconfigより優先されます。

### 繰り返し抑制

```yaml
# 同じ文字・文字列が繰り返されるループを検出した時点でデコードを打ち切り、繰り返しを1回分に縮約するかどうか
repetition_stop: bool

# 繰り返しとみなす単位の最大長（トークン数）
rep_period_max: int

# 1文字の繰り返しを抑制するのに必要な最小連続回数
rep_min_run_p1: int

# 2文字以上の単位の繰り返しを抑制するのに必要な最小繰り返し回数
rep_min_repeats: int
```

デフォルト値は全モデル共通で、`repetition_stop: true`、`rep_period_max: 8`、`rep_min_run_p1: 16`、`rep_min_repeats: 3` です。PyTorch 推論でのみ動作し、ONNX 推論（`infer_onnx=True`）では適用されません。

CLI の `--disable-repetition-stop`、または Python API の `TextRecognizer(repetition_stop=False)` で無効化できます。これらの指定はモデルconfigより優先されます。`DocumentAnalyzer` / `OCR` からの指定方法は [設定の与え方](#how-to-configure) を参照してください。

### 可視化設定

```yaml
visualize:
  # 予測結果文字列の可視化に用いるフォントのパス
  font: str

  # 予測結果文字列の可視化に用いるフォントの色
  color: [BGR]

  # 予測結果文字列のフォントの大きさ
  font_size: int
```

## Layout Parser

### 予測スコアに対する閾値

```yaml
# モデルの予測スコアに対する閾値で、予測スコアが設定した閾値を領域を除外します。
thresh_score: float 
```

## Table Structure Recognizer

### 予測スコアに対する閾値

```yaml
# モデルの予測スコアに対する閾値で、予測スコアが設定した閾値を領域を除外します。
thresh_score: float 
```
