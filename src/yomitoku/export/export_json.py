import json

import cv2
import os


def paragraph_to_json(paragraph, ignore_line_break):
    if ignore_line_break:
        paragraph.contents = paragraph.contents.replace("\n", "")


def table_to_json(table, ignore_line_break):
    for cell in table.cells:
        if ignore_line_break:
            cell.contents = cell.contents.replace("\n", "")


def save_figure(
    figures,
    img,
    out_path,
    figure_dir="figures",
):
    assert img is not None, "img is required for saving figures"

    for i, figure in enumerate(figures):
        x1, y1, x2, y2 = map(int, figure.box)
        figure_img = img[y1:y2, x1:x2, :]
        save_dir = os.path.dirname(out_path)
        save_dir = os.path.join(save_dir, figure_dir)
        os.makedirs(save_dir, exist_ok=True)

        filename = os.path.splitext(os.path.basename(out_path))[0]
        figure_name = f"{filename}_figure_{i}.png"
        figure_path = os.path.join(save_dir, figure_name)
        cv2.imwrite(figure_path, figure_img)


def export_json(
    inputs,
    out_path,
    ignore_line_break=False,
    encoding: str = "utf-8",
    img=None,
    export_figure=False,
    figure_dir="figures",
):
    from yomitoku.document_analyzer import DocumentAnalyzerSchema

    if isinstance(inputs, DocumentAnalyzerSchema):
        for table in inputs.tables:
            table_to_json(table, ignore_line_break)

    if isinstance(inputs, DocumentAnalyzerSchema):
        for paragraph in inputs.paragraphs:
            paragraph_to_json(paragraph, ignore_line_break)

        if export_figure:
            save_figure(
                inputs.figures,
                img,
                out_path,
                figure_dir=figure_dir,
            )

    with open(out_path, "w", encoding=encoding, errors="ignore") as f:
        json.dump(
            inputs.model_dump(),
            f,
            ensure_ascii=False,
            indent=4,
            sort_keys=True,
            separators=(",", ": "),
        )
