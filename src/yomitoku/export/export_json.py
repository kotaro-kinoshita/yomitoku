import json
import os

from ..utils.misc import save_image


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
        save_image(figure_img, figure_path)


def convert_json(inputs, out_path, ignore_line_break, img, export_figure, figure_dir):
    from yomitoku.document_analyzer import DocumentAnalyzerSchema

    if isinstance(inputs, DocumentAnalyzerSchema):
        for table in inputs.tables:
            table_to_json(table, ignore_line_break)

    if isinstance(inputs, DocumentAnalyzerSchema):
        for paragraph in inputs.paragraphs:
            paragraph_to_json(paragraph, ignore_line_break)

    if isinstance(inputs, DocumentAnalyzerSchema) and export_figure:
        save_figure(
            inputs.figures,
            img,
            out_path,
            figure_dir=figure_dir,
        )

    return inputs


def export_json(
    inputs,
    out_path,
    ignore_line_break=False,
    encoding: str = "utf-8",
    img=None,
    export_figure=False,
    figure_dir="figures",
):
    inputs = convert_json(
        inputs,
        out_path,
        ignore_line_break,
        img,
        export_figure,
        figure_dir,
    )

    save_json(
        inputs.model_dump(),
        out_path,
        encoding,
    )

    return inputs


def save_json(data, out_path, encoding):
    with open(out_path, "w", encoding=encoding, errors="ignore") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=4,
            sort_keys=True,
            separators=(",", ": "),
        )
