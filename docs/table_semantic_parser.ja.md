# TableSemanticParser 使い方（解析・可視化・テンプレート）

## 1. 目的

`TableSemanticParser` は PDF/画像から

* OCR結果（文字）
* テーブル検出（セルbbox）
* テーブルの意味解析（Key-Value / Grid）

を推定し、結果を **JSON/CSV** に保存したり、検索・可視化できます。

さらに、解析結果を **テンプレートJSON** として保存し、外部編集した内容を次回の解析へ適用できます。

---

## 2. 基本の解析フロー

### 2.1 サンプル（PDFをページごとに解析）

```python
import cv2
import os

from yomitoku.table_semantic_parser import TableSemanticParser
from yomitoku.data.functions import load_pdf

analyzer = TableSemanticParser(
    device="cuda:1",
    visualize=True,
)

path_img = "dataset/EPSON012.PDF"
imgs = load_pdf(path_img)
output = "outputs"
os.makedirs(output, exist_ok=True)

basename = os.path.splitext(os.path.basename(path_img))[0]

for i, img in enumerate(imgs):
    results, vis_cell, vis_ocr = analyzer(img)

    if len(results.tables) == 0:
        print(f"No table found in {path_img} page {i}")
        continue

    cv2.imwrite(f"{output}/{basename}_{i}_ocr.jpg", vis_ocr)     # OCR可視化
    cv2.imwrite(f"{output}/{basename}_{i}_cell.jpg", vis_cell)   # セル可視化

    results.to_json(f"{output}/{basename}_raw_result_{i}.json")  # 解析結果(JSON)
```

### 2.2 返り値

`analyzer(img)` は以下を返します。

* `results`: `TableSemanticParserSchema`（解析結果本体）
* `vis_cell`: セル検出可視化画像（ndarray）
* `vis_ocr`: OCR可視化画像（ndarray）

---

## 3. Key-Value の検索・出力

### 3.1 ドキュメント全体から Key を検索

```python
address_kv = results.search_kv_items_by_key(key="住所")
print(address_kv)
```

戻り値は「該当KeyセルとValueセルのリスト」です。

---

### 3.2 特定テーブルから検索（table_id指定）

```python
table = results.find_table_by_id(table_id="t0")
print(table.search_kv_items_by_key(key="申込日"))
```

---

### 3.3 KVを辞書形式で取得（閲覧用）

```python
print(table.view.kv_items_to_dict())
```

* keyセルの内容を連結してキー名を作り、valueセルの内容を値にします
* 同名キーが複数ある場合、内部で `_0`, `_1` のようにユニーク化されます（例：`フリガナ_0`）

---

### 3.4 KVをJSON/CSVに保存

```python
table.export.kv_items_to_json(f"{output}/{basename}_kv_items_{i}.json")
table.export.kv_items_to_csv(f"{output}/{basename}_kv_items_{i}.csv")
```

---

## 4. セル検索（隣接 / bbox / クエリ）

### 4.1 Key文字列の右隣セル

```python
table = results.find_table_by_id(table_id="t1")
print(table.search_cells_right_of_key_text(key="イベント名"))
```

### 4.2 bbox内のセル検索

```python
print(table.search_cells_by_bbox([380, 1045, 1513, 1165])
```

---

## 5. Grid（明細テーブル）の出力

### 5.1 JSON保存

```python
table = results.find_table_by_id(table_id="t2")
table.export.grids_to_json(f"{output}/{basename}_grids_{i}.json")
```

### 5.2 DataFrame化

```python
df = table.export.grids_to_dataframe()
print(df.head())
```

### 5.3 CSV出力（列指定）

```python
table.export.grids_to_csv(
    columns=["使用日", "開始時間", "終了時間"],
    out_path=f"{output}/{basename}_table_{i}.csv",
)
```

---

## 6. テンプレート（Template JSON）

### 6.1 何のためのテンプレか

テンプレートJSONは **解析結果の“上書き用”定義**です。

* セルの `role` や `contents` を外部で直したい
* KVリンクやGrid構造を外部で固定したい
* 次回解析時にテンプレートを当てて、結果を安定化させたい

といった用途で使います。

---

### 6.2 テンプレートJSONの保存

```python
results.save_template_json(f"{output}/{basename}_template_{i}.json")
```

生成されるテンプレは以下を含みます。

* `meta`: テンプレメタ情報（`match_policy` 等）
* `tables`: テーブル単位のテンプレ

  * `box`: テーブルbbox（適用対象を探すためのキー）
  * `cells`: セル差分（role/contents）
  * `kv_items`: KV定義（任意：`include_kv=True` のとき）
  * `grids`: Grid定義（任意：`include_grids=True` のとき）

※ `save_template_json` では `role == "group"` のセルはテンプレ出力から除外します。

---

### 6.3 テンプレートJSONの編集

テンプレ編集の基本は次のどちらかです。

* **セルの role/contents を編集**

  * “このセルは header にしたい”
  * “OCR誤りをテンプレ側で直したい”
* **KVやGridを固定したいなら kv_items / grids を編集**

#### KVの書式（読みやすいJSON）

KVの `key/value` は、**1つなら文字列、複数ならリスト**の両方が使えます。

```json
{ "key": "c12", "value": "c21" }
{ "key": "c30", "value": ["c31", "c32"] }
```

内部では常に `List[str]` として扱われ、出力時に1要素なら文字列に戻ります。

#### Gridの書式（row_keys/col_keys）

`row_keys` / `col_keys` は同じく **文字列 or リスト**を許容します。

```json
{ "row_keys": "c1", "col_keys": ["c2","c3"], "value": "c10" }
```

---

### 6.4 テンプレを使って再解析する

テンプレは `analyzer(..., template=...)` で渡します。

```python
results, vis_cell, vis_ocr = analyzer(
    img,
    template=f"{output}/{basename}_template_{i}.json",
)
```

テンプレ適用後の挙動：

* テーブルはテンプレtableの `box` と、解析結果tableの位置を比較して対応付けします（overlap ratio）
* セルの上書きは `meta.match_policy` で決まります

  * `cell_id`: cell_id一致
  * `bbox`: bbox一致（包含判定）
* `kv_items` / `grids` がテンプレに存在する場合は、解析結果のそれらを **丸ごと差し替え**ます

---

## 7. 運用のベストプラクティス

### 7.1 まずはセルrole/contentsだけテンプレ化

最初から `kv_items/grids` まで固定すると編集量が増えるので、

1. `cells` の role/contents を修正してテンプレ適用を安定させる
2. 必要になったら `kv_items` を固定
3. 最後に `grids` を固定

が楽です。

### 7.2 match_policy の選び方

* **基本は `cell_id` 推奨**（再現性が高い）
* `bbox` は、セルIDが変動するケース（再生成される等）に使用

---

## 8. まとめ：最小のテンプレ運用手順

1. まず解析して `*_cell.jpg` を見ながらテンプレJSONを保存
2. 外部でテンプレJSON（cells/kv/grids）を編集
3. `template=...` を渡して再解析
4. 結果のKV/CSV出力を下流処理へ
