# Module Schema

## Document Analyzer

Document Analyzer モジュールは以下の変数を `tuple` で出力します。

| 変数名 | 型 | 説明 |
| :--- | :--- | :--- |
| results | `DocumentAnalyzerSchema` | モジュールの出力結果 |
| ocr_vis | `np.array` \| `None` | OCR モジュールの可視化（`visualizer=True` の時のみ） |
| layout_vis | `np.array` \| `None` | Layout Analyzer モジュールの可視化（`visualizer=True` の時のみ） |

`results` 変数の準拠するスキーマ `DocumentAnalyzerSchema` の仕様は以下の通りです。

{{ schema_cards("document_analyzer_schema") }}

## AI-OCR

AI-OCR モジュールは以下の変数を `tuple` で出力します。

| 変数名 | 型 | 説明 |
| :--- | :--- | :--- |
| results | `OCRSchema` | モジュールの出力結果 |
| ocr_vis | `np.array` \| `None` | AI-OCR モジュールの可視化（`visualizer=True`の時のみ） |

`results` 変数の準拠するスキーマ `OCRSchema` の仕様は以下の通りです。

{{ schema_cards("ocr_schema") }}

## Layout Analyzer

Layout Analyzer モジュールは以下の変数を `tuple` で出力します。

| 変数名 | 型 | 説明 |
| :--- | :--- | :--- |
| results | `LayoutAnalyzerSchema` | モジュールの出力結果 |
| layout_vis | `np.array` \| `None` | Layout Analyzer モジュールの可視化（`visualizer=True`の時のみ） |

`results` 変数の準拠するスキーマ `LayoutAnalyzerSchema` の仕様は以下の通りです。

{{ schema_cards("layout_analyzer_schema") }}

---

_Auto-generated from JSON Schema files._
