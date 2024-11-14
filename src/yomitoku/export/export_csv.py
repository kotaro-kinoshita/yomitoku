import csv


def table_to_csv(table):
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
                if i == row and j == col:
                    table_array[i][j] = contents

    return table_array


def export_csv(inputs, out_path: str):
    elements = []
    for table in inputs.tables:
        table_md = table_to_csv(table)

        elements.append(
            {
                "type": "table",
                "box": table.box,
                "element": table_md,
            }
        )

    for paraghraph in inputs.paragraphs:
        contents = paraghraph.contents
        elements.append(
            {
                "type": "paragraph",
                "box": paraghraph.box,
                "element": contents,
            }
        )

    elements = sorted(elements, key=lambda x: x["box"][1])

    with open(out_path, "w") as f:
        writer = csv.writer(f)
        for element in elements:
            if element["type"] == "table":
                writer.writerows(element["element"])
            else:
                writer.writerow([element["element"]])

            writer.writerow([""])
