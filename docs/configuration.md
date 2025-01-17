# Configuration

核モジュールに対して、設定可能なパラメータについて説明します。

## Text Detector

### 入力画像サイズに関する設定

```yaml
data:
  shortest_size: int # 画像の短辺ピクセル数が設定した数値を下回る場合にここで設定した画像のピクセル数に以上になるように画像を拡大します。
  limit_size: int #画像の長辺ピクセル数が設定した数値を上回る場合にここで設定した画像のピクセル数以下になるように画像を縮小します。
```

### 後処理

```yaml
post_process:
  min_size: int #検出した領域の辺の大きいさが設定した数値を下回る場合に領域を除去します。
  thresh: float #モデルの予測スコアに対する閾値で、予測スコアが設定した閾値を下回るピクセルを背景領域として扱います。
  box_thresh: float #領域内の予測の平均スコアに対する閾値で、閾値を下回る領域を除外する
  max_candidates: int #検出可能なテキスト領域数の上限
  unclip_ratio: int #テキスト領域のマージン領域の大きさを設定するためのパラメータ。大きいほど、テキスト領域のマージンを大きくし、余白を持たせた検出が可能になり、小さいほどタイトな検出になる。
```

### 可視化設定

```yaml
visualize:
  color: [B, G, R] #検出領域のバウンディングボックスの色の設定
  heatmap: boolean #モデルの予測ヒートマップを可視化、描画するか
```

## Text Recognizer

### 文字列長

```yaml
max_label_length: int #予測可能な最大文字列長
```

### 入力画像

```yaml
data:
  batch_size: int #バッチ処理に用いる画像数
```

### 可視化設定

```yaml
visualize:
  font: str # 予測結果文字列の可視化に用いるフォントのパス
  color: [BGR] # 予測結果文字列の可視化に用いるフォントの色
  font_size: int # 予測結果文字列のフォントの大きさ
```

## Layout_parser

### 予測スコアに対する閾値

```yaml
thresh_score: float #モデルの予測スコアに対する閾値で、予測スコアが設定した閾値を領域を除外します。
```

## Table Structure Recognizer

### 予測スコアに対する閾値

```yaml
thresh_score: float #モデルの予測スコアに対する閾値で、予測スコアが設定した閾値を領域を除外します。
```
