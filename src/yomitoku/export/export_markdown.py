def escape_markdown_text(text):
    specific_chars = [
        "_",
        "*",
        "[",
        "]",
        "(",
        ")",
        "`",
        "#",
        "+",
        "-",
        "|",
        "{",
        "}",
        ".",
        "!",
    ]
    for char in specific_chars:
        text = text.replace(char, "\\" + char)

    return text


def table_to_md(table):
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
                contents = contents.replace("\n", "<br>")
                contents = escape_markdown_text(contents)

                if i == row and j == col:
                    table_array[i][j] = contents

    table_md = ""
    for i in range(num_rows):
        row = "|".join(table_array[i])
        table_md += f"|{row}|\n"

        if i == 0:
            header = "|".join(["-" for _ in range(num_cols)])
            table_md += f"|{header}|\n"

    return table_md


def export_markdown(inputs, out_path: str):
    elements = []
    for table in inputs.tables:
        table_md = table_to_md(table)
        elements.append(
            {
                "box": table.box,
                "md": table_md,
            }
        )

    for paraghraph in inputs.paragraphs:
        contents = escape_markdown_text(paraghraph.contents)

        elements.append(
            {
                "box": paraghraph.box,
                "md": contents + "\n",
            }
        )

    elements = sorted(elements, key=lambda x: x["box"][1])
    markdonw = "\n".join([element["md"] for element in elements])

    with open(out_path, "w") as f:
        f.write(markdonw)
