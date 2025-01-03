import csv
import cv2
import os


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


def export_csv(
    inputs,
    out_path: str,
    ignore_line_break: bool = False,
    encoding: str = "utf-8",
    img=None,
    export_figure: bool = True,
    figure_dir="figures",
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

    if export_figure:
        save_figure(
            inputs.figures,
            img,
            out_path,
            figure_dir=figure_dir,
        )

    elements = sorted(elements, key=lambda x: x["order"])

    with open(out_path, "w", newline="", encoding=encoding, errors="ignore") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        for element in elements:
            if element["type"] == "table":
                writer.writerows(element["element"])
            else:
                writer.writerow([element["element"]])

            writer.writerow([""])
