# スキーマ自動生成プロンプト

以下のプロンプトをLLM（ChatGPT、Claude等）に入力すると、Yomitoku Extractorのスキーマ定義YAMLを自動生成できます。

`{ITEMS}` を取得したい項目（日本語、カンマ区切り）に置き換えて使用してください。

---

## プロンプト

```text
あなたは「Yomitoku Extractor」のスキーマ（YAML）を生成する。出力は YAML のみ。説明文・前置き・コードブロック（```）は禁止。

【入力】
取得項目（日本語、カンマ区切り）:
{ITEMS}

【出力仕様（厳守）】
- ルートは必ず `fields:` から開始する
- YAML はインデント2スペース、タブ禁止
- fields は入力の順番を基本に並べる
- 各フィールドは必ず以下4つを含める:
  - name: snake_case の英小文字（日本語禁止、英数字と _ のみ）
  - structure: kv または table
  - description: 日本語（帳票上のキー/ヘッダーとして検索に使う語。原則は入力項目名を短くしたもの）
  - type: string / number / date / alphanumeric / hiragana / katakana のいずれか
- normalize は必要な場合のみ追加する（不要なら出さない）
- cell_id / bbox / regex は原則出さない（入力に「cell_id=」「bbox=」「regex=」が含まれる場合のみ出す）
- 同義語・重複項目があれば統合して1つにする（description は最も一般的な語に寄せる）

【structure の決定ルール】
- 原則 `kv`
- `table` にできるのは次の条件を両方満たすときだけ:
  1) 取得項目名に「明細」「内訳」「一覧」「表」「リスト」「行」「レコード」「履歴」が含まれる
  2) 同じ取得項目内に列候補が `/` または `()` または `:` で列挙されている
     例: `明細(商品名/数量/単価/金額)` や `利用履歴: 日付/入室/退室`
- table の場合:
  - フィールド側: name / structure: table / description / type を必ず出す（type は string でよい）
  - 必ず `columns:` を作り、2列以上にする
  - columns の各要素は必ず name / description / type を含める（normalize は必要な場合のみ）

【name の生成ルール】
- 日本語の項目名を意味が通る範囲で英語にして snake_case 化する
- 推奨語彙（迷ったらこれを優先）:
  会社名=company_name, 氏名=full_name, 住所=address, 郵便番号=postal_code,
  電話番号=phone_number, FAX=fax_number, メール=email, 担当者=person_in_charge,
  請求書番号=invoice_number, 注文番号=order_number, 登録番号=registration_number,
  発行日=issue_date, 請求日=billing_date, 納品日=delivery_date, 支払期限=due_date,
  小計=subtotal, 税額=tax_amount, 合計=total_amount, 合計金額=total_amount,
  数量=quantity, 単価=unit_price, 金額=amount, 商品名=product, 品名=item_name,
  明細=items, 内訳=breakdown, 一覧=list, 備考=notes
- 推奨語彙にない場合も必ず英語にして name を作る（ローマ字は禁止）
- table の columns 内の name も同じルールに従う

【type / normalize の推定ルール】
以下のキーワードが項目名に含まれる場合、対応する type / normalize を適用する。
上から順に評価し、最初にマッチしたルールを使う。

| キーワード | type | normalize | 備考 |
|---|---|---|---|
| 電話/TEL/FAX | string | phone_jp | ハイフン区切り電話番号に正規化 |
| 郵便/〒 | string | postal_code_jp | 123-4567 形式に正規化 |
| 日付/年月日/発行日/請求日/納品日/期限 | date | date_jp | YYYY-MM-DD 形式に変換。YYYYMMDD が必要なら date_yyyymmdd |
| 時刻/入室/退室/開始時間/終了時間 | string | time_hms | HH:MM:SS 形式に変換。X時XX分 形式が必要なら time_jp |
| 金額/合計/小計/税額/単価/数量/円/¥ | number | numeric | 数字以外を除去 |
| 住所/所在地 | string | strip_spaces | 空白を除去 |
| 会社名/氏名/名前/名称/施設名/備考 | string | strip_spaces | 空白を除去 |
| 番号/ID/コード（電話・郵便を除く） | alphanumeric | alphanumeric | 半角英数字のみ抽出 |
| ひらがな/ふりがな/かな | hiragana | hiragana | カタカナ→ひらがな変換 |
| カタカナ/フリガナ/カナ | katakana | katakana | ひらがな→カタカナ変換 |
| メール/email | string | （なし） | normalize は付けない |
| 上記のいずれにも該当しない | string | （なし） | normalize は付けない |

【利用可能な normalize 一覧（これ以外は使用禁止）】
strip_spaces, numeric, phone_jp, postal_code_jp,
date_jp, date_yyyymmdd, time_jp, time_hms,
alphanumeric, hiragana, katakana

【出力例】

入力: 会社名, 電話番号, 請求日, 合計金額, 明細(商品名/数量/金額)

fields:
  - name: company_name
    structure: kv
    description: 会社名
    type: string
    normalize: strip_spaces
  - name: phone_number
    structure: kv
    description: 電話番号
    type: string
    normalize: phone_jp
  - name: billing_date
    structure: kv
    description: 請求日
    type: date
    normalize: date_jp
  - name: total_amount
    structure: kv
    description: 合計金額
    type: number
    normalize: numeric
  - name: items
    structure: table
    description: 明細
    type: string
    columns:
      - name: product
        description: 商品名
        type: string
      - name: quantity
        description: 数量
        type: number
        normalize: numeric
      - name: amount
        description: 金額
        type: number
        normalize: numeric

以上に従い、{ITEMS} から YAML を生成して出力せよ。
```

---

## 使用例

**入力:**

```
契約者名, 住所, 電話番号, 郵便番号, 利用希望日(日付/入室/退室), 備考
```

**生成される YAML:**

```yaml
fields:
  - name: contractor_name
    structure: kv
    description: 契約者名
    type: string
    normalize: strip_spaces
  - name: address
    structure: kv
    description: 住所
    type: string
    normalize: strip_spaces
  - name: phone_number
    structure: kv
    description: 電話番号
    type: string
    normalize: phone_jp
  - name: postal_code
    structure: kv
    description: 郵便番号
    type: string
    normalize: postal_code_jp
  - name: usage_schedule
    structure: table
    description: 利用希望日
    type: string
    columns:
      - name: date
        description: 日付
        type: date
        normalize: date_jp
      - name: check_in_time
        description: 入室
        type: string
        normalize: time_hms
      - name: check_out_time
        description: 退室
        type: string
        normalize: time_hms
  - name: notes
    structure: kv
    description: 備考
    type: string
    normalize: strip_spaces
```
