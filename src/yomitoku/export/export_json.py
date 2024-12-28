import json


def paragraph_to_json(paragraph, ignore_line_break):
    if ignore_line_break:
        paragraph.contents = paragraph.contents.replace("\n", "")


def table_to_json(table, ignore_line_break):
    for cell in table.cells:
        if ignore_line_break:
            cell.contents = cell.contents.replace("\n", "")


def export_json(
    inputs,
    out_path,
    ignore_line_break=False,
    encoding: str = "utf-8",
):
    from yomitoku.document_analyzer import DocumentAnalyzerSchema

    if isinstance(inputs, DocumentAnalyzerSchema):
        for table in inputs.tables:
            table_to_json(table, ignore_line_break)

    if isinstance(inputs, DocumentAnalyzerSchema):
        for paragraph in inputs.paragraphs:
            paragraph_to_json(paragraph, ignore_line_break)

    with open(out_path, "w", encoding=encoding) as f:
        json.dump(
            inputs.model_dump(),
            f,
            ensure_ascii=False,
            indent=4,
            sort_keys=True,
            separators=(",", ": "),
        )
