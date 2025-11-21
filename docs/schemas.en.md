# Module Schema

## Document Analyzer

The Document Analyzer Module outputs the following variables as a `tuple`.

| Variable Name | Type | Description |
| :--- | :--- | :--- |
| results | `DocumentAnalyzerSchema` | Module output results |
| ocr_vis | `ndarray` \| `None` | Visualization of the OCR module (Only when `visualizer=True`) |
| layout_vis | `ndarray` \| `None` | Visualization of the Layout Analyzer module (Only when `visualizer=True`) |

The specification for the `DocumentAnalyzerSchema` that the `results` variable conforms to is as follows:

{{ schema_cards("document_analyzer_schema") }}

---

## AI-OCR

The AI-OCR module outputs the following variables as a `tuple`.

| Variable Name | Type | Description |
| :--- | :--- | :--- |
| results | `OCRSchema` | Module output results |
| ocr_vis | `ndarray` \| `None` | Visualization of the AI-OCR module (Only when `visualizer=True`) |

The specification for the `OCRSchema` that the `results` variable conforms to is as follows:

{{ schema_cards("ocr_schema") }}

---

## Layout Analyzer

The Layout Analyzer module outputs the following variables as a `tuple`.

| Variable Name | Type | Description |
| :--- | :--- | :--- |
| results | `LayoutAnalyzerSchema` | Module output results |
| layout_vis | `ndarray` \| `None` | Visualization of the Layout Analyzer module (Only when `visualizer=True`) |

The specification for the `LayoutAnalyzerSchema` that the `results` variable conforms to is as follows:

{{ schema_cards("layout_analyzer_schema") }}

---

_Auto-generated from JSON Schema files._
