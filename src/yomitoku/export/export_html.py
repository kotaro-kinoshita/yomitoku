import os
import re
from html import escape

import cv2
from lxml import etree, html


def convert_text_to_html(text):
    """
    入力されたテキストをHTMLに変換する関数。
    URLを検出してリンク化せずそのまま表示し、それ以外はHTMLエスケープする。
    """
    url_regex = re.compile(r"https?://[^\s<>]")

    def replace_url(match):
        url = match.group(0)
        return escape(url)

    return url_regex.sub(replace_url, escape(text))


def add_td_tag(contents, row_span, col_span):
    return f'<td rowspan="{row_span}" colspan="{col_span}">{contents}</td>'


def add_table_tag(contents):
    return f'<table border="1" style="border-collapse: collapse">{contents}</table>'


def add_tr_tag(contents):
    return f"<tr>{contents}</tr>"


def add_p_tag(contents):
    return f"<p>{contents}</p>"


def add_html_tag(text):
    return f"<html><body>{text}</body></html>"


def add_h1_tag(contents):
    return f"<h1>{contents}</h1>"


def table_to_html(table, ignore_line_break):
    pre_row = 1
    rows = []
    row = []
    for cell in table.cells:
        if cell.row != pre_row:
            rows.append(add_tr_tag("".join(row)))
            row = []

        row_span = cell.row_span
        col_span = cell.col_span
        contents = cell.contents

        if contents is None:
            contents = ""

        contents = convert_text_to_html(contents)

        if ignore_line_break:
            contents = contents.replace("\n", "")
        else:
            contents = contents.replace("\n", "<br>")

        row.append(add_td_tag(contents, row_span, col_span))
        pre_row = cell.row
    else:
        rows.append(add_tr_tag("".join(row)))

    table_html = add_table_tag("".join(rows))

    return {
        "box": table.box,
        "order": table.order,
        "html": table_html,
    }


def paragraph_to_html(paragraph, ignore_line_break):
    contents = paragraph.contents
    contents = convert_text_to_html(contents)

    if ignore_line_break:
        contents = contents.replace("\n", "")
    else:
        contents = contents.replace("\n", "<br>")

    if paragraph.role == "section_headings":
        contents = add_h1_tag(contents)

    return {
        "box": paragraph.box,
        "order": paragraph.order,
        "html": add_p_tag(contents),
    }


def figure_to_html(
    figures,
    img,
    out_path,
    export_figure_letter=False,
    ignore_line_break=False,
    figure_dir="figures",
    width=200,
):
    assert img is not None, "img is required for saving figures"

    elements = []
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

        elements.append(
            {
                "order": figure.order,
                "html": f'<img src="{figure_dir}/{figure_name}" width="{width}"><br>',
            }
        )

        if export_figure_letter:
            paragraphs = sorted(figure.paragraphs, key=lambda x: x.order)
            for paragraph in paragraphs:
                contents = paragraph_to_html(paragraph, ignore_line_break)
                html = contents["html"]
                elements.append(
                    {
                        "order": figure.order,
                        "html": html,
                    }
                )

    return elements


def export_html(
    inputs,
    out_path: str,
    ignore_line_break: bool = False,
    export_figure: bool = True,
    export_figure_letter: bool = False,
    img=None,
    figure_width=200,
    figure_dir="figures",
    encoding: str = "utf-8",
):
    html_string = ""
    elements = []
    for table in inputs.tables:
        elements.append(table_to_html(table, ignore_line_break))

    for paragraph in inputs.paragraphs:
        elements.append(paragraph_to_html(paragraph, ignore_line_break))

    if export_figure:
        elements.extend(
            figure_to_html(
                inputs.figures,
                img,
                out_path,
                export_figure_letter,
                ignore_line_break,
                width=figure_width,
                figure_dir=figure_dir,
            ),
        )

    elements = sorted(elements, key=lambda x: x["order"])

    html_string = "".join([element["html"] for element in elements])
    html_string = add_html_tag(html_string)

    parsed_html = html.fromstring(html_string)
    formatted_html = etree.tostring(parsed_html, pretty_print=True, encoding="unicode")

    with open(out_path, "w", encoding=encoding, errors="ignore") as f:
        f.write(formatted_html)

    return formatted_html
