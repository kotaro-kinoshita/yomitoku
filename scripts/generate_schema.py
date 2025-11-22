import json
from typing import Any, Dict

from yomitoku.schemas import DocumentAnalyzerSchema, LayoutAnalyzerSchema, OCRSchema


def output_json(targets: list[Dict[str, Any]]):
    """output json schema files"""

    for target in targets:
        schema = target["schema"].model_json_schema()
        outpath = f"schemas/{target['name']}.json"

        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)


def main() -> None:
    targets = [
        {"schema": DocumentAnalyzerSchema, "name": "document_analyzer_schema"},
        {"schema": OCRSchema, "name": "ocr_schema"},
        {"schema": LayoutAnalyzerSchema, "name": "layout_analyzer_schema"},
    ]

    output_json(targets)


if __name__ == "__main__":
    main()
