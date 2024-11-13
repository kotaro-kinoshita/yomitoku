from xml.dom.minidom import parseString


def export_html(outputs):
    html = ""

    elements = []
    for table in outputs.tables:
        table_html = (
            """<table border='1' style='border-collapse: collapse'><tr>"""
        )

        pre_row = 1
        for cell in table.cells:
            row = cell.row
            row_span = cell.row_span
            col_span = cell.col_span
            contents = cell.contents
            if contents is None:
                contents = ""

            if row != pre_row:
                table_html += "</tr><tr>"
                pre_row = row

            table_html += f'<td rowspan="{row_span}" colspan="{col_span}">{contents}</td>'
        else:
            table_html += "</tr></table>"

        elements.append(
            {
                "box": table.box,
                "html": table_html,
            }
        )

    for paraghraph in outputs.paragraphs:
        p = f"<p>{paraghraph.contents}</p>"
        elements.append(
            {
                "box": paraghraph.box,
                "html": p,
            }
        )

    elements = sorted(elements, key=lambda x: x["box"][1])
    html = "".join([element["html"] for element in elements])
    html = add_html_tag(html)
    html = parseString(html).toprettyxml(indent="  ")

    with open("output.html", "w") as f:
        f.write(html)

    return html


def add_html_tag(text):
    return f"<html><body>{text}</body></html>"
