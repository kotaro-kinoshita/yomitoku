from yomitoku.schemas import DocumentAnalyzerSchema, OCRSchema, LayoutAnalyzerSchema
from json_schema_for_humans.generate import generate_from_filename

import json

targets = [
    {"schema": DocumentAnalyzerSchema, "name": "document_analyzer_schema"},
    {"schema": OCRSchema, "name": "ocr_schema"},
    {"schema": LayoutAnalyzerSchema, "name": "layout_analyzer_schema"},
]

for target in targets:
    schema = target["schema"].model_json_schema()
    outpath = f"schemas/{target['name']}.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)

    # Generate HTML documentation for the schemas
    generate_from_filename(
        f"schemas/{target['name']}.json", f"docs/{target['name']}.md"
    )
