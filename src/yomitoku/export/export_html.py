import re
from html import escape

from lxml import etree, html

from ..utils.reading_order import reading_order
from .utils import sort_elements


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
        "html": table_html,
    }


def paragraph_to_html(paragraph, ignore_line_break):
    contents = paragraph.contents
    contents = convert_text_to_html(contents)

    if ignore_line_break:
        contents = contents.replace("\n", "")
    else:
        contents = contents.replace("\n", "<br>")

    return {
        "box": paragraph.box,
        "html": add_p_tag(contents),
    }


def export_html(
    inputs,
    out_path: str,
    ignore_line_break: bool = False,
):
    html_string = ""
    elements = []
    for table in inputs.tables:
        elements.append(table_to_html(table, ignore_line_break))

    for paraghraph in inputs.paragraphs:
        elements.append(paragraph_to_html(paraghraph, ignore_line_break))

    # directions = [paraghraph.direction for paraghraph in inputs.paragraphs]
    # elements = sort_elements(elements, directions)

    order = reading_order(elements)
    elements = [elements[i] for i in order]

    html_string = "".join([element["html"] for element in elements])
    html_string = add_html_tag(html_string)

    parsed_html = html.fromstring(html_string)
    formatted_html = etree.tostring(
        parsed_html, pretty_print=True, encoding="unicode"
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(formatted_html)
