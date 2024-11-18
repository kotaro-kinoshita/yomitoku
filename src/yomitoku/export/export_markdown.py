import re

from .utils import sort_elements


def escape_markdown_special_chars(text):
    special_chars = r"([`*_{}[\]()#+.!|-])"
    return re.sub(special_chars, r"\\\1", text)


def paragraph_to_md(paragraph, ignore_line_break):
    contents = escape_markdown_special_chars(paragraph.contents)

    if ignore_line_break:
        contents = contents.replace("\n", "")
    else:
        contents = contents.replace("\n", "<br>")

    return {
        "box": paragraph.box,
        "md": contents + "\n",
    }


def table_to_md(table, ignore_line_break):
    num_rows = table.n_row
    num_cols = table.n_col

    table_array = [["" for _ in range(num_cols)] for _ in range(num_rows)]

    for cell in table.cells:
        row = cell.row - 1
        col = cell.col - 1
        row_span = cell.row_span
        col_span = cell.col_span
        contents = cell.contents

        for i in range(row, row + row_span):
            for j in range(col, col + col_span):
                contents = escape_markdown_special_chars(contents)
                if ignore_line_break:
                    contents = contents.replace("\n", "")
                else:
                    contents = contents.replace("\n", "<br>")

                if i == row and j == col:
                    table_array[i][j] = contents

    table_md = ""
    for i in range(num_rows):
        row = "|".join(table_array[i])
        table_md += f"|{row}|\n"

        if i == 0:
            header = "|".join(["-" for _ in range(num_cols)])
            table_md += f"|{header}|\n"

    return {
        "box": table.box,
        "md": table_md,
    }


def export_markdown(inputs, out_path: str, ignore_line_break: bool = False):
    elements = []
    for table in inputs.tables:
        elements.append(table_to_md(table, ignore_line_break))

    for paraghraph in inputs.paragraphs:
        elements.append(paragraph_to_md(paraghraph, ignore_line_break))

    directions = [paraghraph.direction for paraghraph in inputs.paragraphs]
    sort_elements(elements, directions)

    markdonw = "\n".join([element["md"] for element in elements])

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(markdonw)
