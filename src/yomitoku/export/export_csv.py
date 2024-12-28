import csv


def table_to_csv(table, ignore_line_break):
    num_rows = table.n_row
    num_cols = table.n_col

    table_array = [["" for _ in range(num_cols)] for _ in range(num_rows)]

    for cell in table.cells:
        row = cell.row - 1
        col = cell.col - 1
        row_span = cell.row_span
        col_span = cell.col_span
        contents = cell.contents

        if ignore_line_break:
            contents = contents.replace("\n", "")

        for i in range(row, row + row_span):
            for j in range(col, col + col_span):
                if i == row and j == col:
                    table_array[i][j] = contents
    return table_array


def paragraph_to_csv(paragraph, ignore_line_break):
    contents = paragraph.contents

    if ignore_line_break:
        contents = contents.replace("\n", "")

    return contents


def export_csv(
    inputs,
    out_path: str,
    ignore_line_break: bool = False,
    encoding: str = "utf-8",
):
    elements = []
    for table in inputs.tables:
        table_csv = table_to_csv(table, ignore_line_break)

        elements.append(
            {
                "type": "table",
                "box": table.box,
                "element": table_csv,
                "order": table.order,
            }
        )

    for paraghraph in inputs.paragraphs:
        contents = paragraph_to_csv(paraghraph, ignore_line_break)
        elements.append(
            {
                "type": "paragraph",
                "box": paraghraph.box,
                "element": contents,
                "order": paraghraph.order,
            }
        )

    elements = sorted(elements, key=lambda x: x["order"])

    with open(out_path, "w", newline="", encoding=encoding) as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        for element in elements:
            if element["type"] == "table":
                writer.writerows(element["element"])
            else:
                writer.writerow([element["element"]])

            writer.writerow([""])
