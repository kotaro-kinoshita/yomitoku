import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, features
from ..constants import PALETTE, ROOT_DIR
from .logger import set_logger

logger = set_logger(__name__, "INFO")


def _reading_order_visualizer(img, elements, line_color, tip_size):
    out = img.copy()
    for i, element in enumerate(elements):
        cur_x1, cur_y1, cur_x2, cur_y2 = element.box

        cur_center = (
            cur_x1 + (cur_x2 - cur_x1) / 2,
            cur_y1 + (cur_y2 - cur_y1) / 2,
        )

        cv2.putText(
            out,
            str(i),
            (int(cur_center[0]), int(cur_center[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 200, 0),
            2,
        )

        if i == 0:
            continue

        prev_element = elements[i - 1]
        prev_x1, prev_y1, prev_x2, prev_y2 = prev_element.box
        prev_center = (
            prev_x1 + (prev_x2 - prev_x1) / 2,
            prev_y1 + (prev_y2 - prev_y1) / 2,
        )

        arrow_length = np.linalg.norm(np.array(cur_center) - np.array(prev_center))

        # tipLength を計算（矢印長さに対する固定サイズの割合）
        if arrow_length > 0:
            tip_length = tip_size / arrow_length
        else:
            tip_length = 0  # 長さが0なら矢じりもゼロ

        cv2.arrowedLine(
            out,
            (int(prev_center[0]), int(prev_center[1])),
            (int(cur_center[0]), int(cur_center[1])),
            line_color,
            2,
            tipLength=tip_length,
        )
    return out


def reading_order_visualizer(
    img,
    results,
    line_color=(0, 0, 255),
    tip_size=10,
    visualize_figure_letter=False,
):
    elements = results.paragraphs + results.tables + results.figures
    elements = sorted(elements, key=lambda x: x.order)

    out = _reading_order_visualizer(img, elements, line_color, tip_size)

    if visualize_figure_letter:
        for figure in results.figures:
            out = _reading_order_visualizer(
                out, figure.paragraphs, line_color=(0, 255, 0), tip_size=5
            )

    return out


def det_visualizer(img, quads, preds=None, vis_heatmap=False, line_color=(0, 255, 0)):
    out = img.copy()
    h, w = out.shape[:2]
    if vis_heatmap:
        preds = preds["binary"][0]
        binary = preds.detach().cpu().numpy()
        binary = binary.squeeze(0)
        binary = (binary * 255).astype(np.uint8)
        binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_LINEAR)
        heatmap = cv2.applyColorMap(binary, cv2.COLORMAP_JET)
        out = cv2.addWeighted(out, 0.5, heatmap, 0.5, 0)

    for quad in quads:
        quad = np.array(quad).astype(np.int32)
        out = cv2.polylines(out, [quad], True, line_color, 1)
    return out


def layout_visualizer(results, img):
    out = img.copy()
    results_dict = results
    for id, (category, preds) in enumerate(results_dict.items()):
        for element in preds:
            box = element["box"]
            role = element["role"]

            if category != "tables":
                continue

            if role is None:
                role = ""
            else:
                role = f"({role})"

            color = PALETTE[id % len(PALETTE)]
            x1, y1, x2, y2 = tuple(map(int, box))
            out = cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            out = cv2.putText(
                out,
                category + role,
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    return out


def table_detector_visualizer(img, tables):
    out = img.copy()
    for i, table in enumerate(tables):
        box = table.box
        id = f"t{table.order}"
        x1, y1, x2, y2 = map(int, box)
        out = cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        out = cv2.putText(
            out,
            f"table {id}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 0, 0),
            2,
        )

    return out


def table_visualizer(img, table):
    out = img.copy()
    cells = table.cells
    for cell in cells:
        box = cell.box
        row = cell.row
        col = cell.col
        row_span = cell.row_span
        col_span = cell.col_span

        text = f"[{row}, {col}] ({row_span}x{col_span})"

        x1, y1, x2, y2 = map(int, box)
        out = cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 255), 2)
        out = cv2.putText(
            out,
            text,
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )

    return out


def cell_detector_visualizer(img1, img2, cells):
    out1 = img1.copy()
    out2 = img2.copy()

    fill = np.full_like(img1, 255)
    colors = {
        "cell": (255, 128, 0),
        "empty": (255, 0, 255),
        "header": (0, 255, 0),
        "group": (255, 255, 0),
    }

    for cell in cells:
        box = cell.box
        role = cell.role
        color = colors.get(role, (200, 200, 200))
        if role in ["cell", "header", "empty"]:
            x1, y1, x2, y2 = map(int, box)
            fill = cv2.rectangle(fill, (x1, y1), (x2, y2), color, -1)
            out1 = cv2.rectangle(out1, (x1, y1), (x2, y2), color, 2)

    out1 = np.where(
        fill == 255,
        img1.copy(),
        cv2.addWeighted(img1.copy(), 0.7, fill, 0.3, 0),
    )

    for c in cells:
        box = c.box
        x1, y1, x2, y2 = map(int, box)
        target = out1 if c.role != "group" else out2
        target = cv2.rectangle(target, (x1, y1), (x2, y2), (0, 0, 255), 2)
        target = cv2.putText(
            target,
            c.id,
            (int((x1 + x2) / 2), int((y1 + y2) / 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    return out1, out2


def cell_visualizer(
    img,
    cell,
    font_path=None,
    font_size=12,
    font_color=(255, 0, 0),
    bbox_color=(0, 255, 0),
):
    out = img.copy()
    pillow_img = Image.fromarray(out)
    draw = ImageDraw.Draw(pillow_img)
    has_raqm = features.check_feature(feature="raqm")
    if not has_raqm:
        logger.warning(
            "libraqm is not installed. Vertical text rendering is not supported. Rendering horizontally instead."
        )

    if font_path is None:
        font_path = f"{ROOT_DIR}/resource/MPLUS1p-Medium.ttf"

    font = ImageFont.truetype(font_path, font_size)
    draw.rectangle(
        [tuple(cell.box[0:2]), tuple(cell.box[2:4])],
        outline=bbox_color,
        width=2,
    )
    draw.text(
        (cell.box[0], cell.box[1] - font_size),
        f"{cell.contents}",
        font=font,
        fill=font_color,
    )

    out = np.array(pillow_img)
    return out


def dag_visualizer(dag, img):
    for u, v, attrs in dag.edges(data=True):
        if attrs["dir"] in ["L", "U"]:
            continue
        cx1 = (dag.nodes[u]["bbox"][0] + dag.nodes[u]["bbox"][2]) / 2
        cy1 = (dag.nodes[u]["bbox"][1] + dag.nodes[u]["bbox"][3]) / 2
        cx2 = (dag.nodes[v]["bbox"][0] + dag.nodes[v]["bbox"][2]) / 2
        cy2 = (dag.nodes[v]["bbox"][1] + dag.nodes[v]["bbox"][3]) / 2
        color = (0, 255, 0) if attrs["dir"] == "R" else (255, 0, 0)
        img = cv2.arrowedLine(
            img,
            (int(cx1), int(cy1)),
            (int(cx2), int(cy2)),
            color,
            2,
        )

    return img


def kv_items_visualizer(img, table):
    out = img.copy()
    for kv_item in table.kv_items:
        if len(kv_item.key) == 0 or len(kv_item.value) == 0:
            continue

        start_x = None
        start_y = None
        end_x = None
        end_y = None

        for i, key_cell_id in enumerate(kv_item.key):
            key_cell = table.cells.get(key_cell_id)

            end_x = (key_cell.box[0] + key_cell.box[2]) // 2
            end_y = (key_cell.box[1] + key_cell.box[3]) // 2

            cv2.rectangle(
                out,
                (key_cell.box[0], key_cell.box[1]),
                (key_cell.box[2], key_cell.box[3]),
                color=(0, 255, 0),
                thickness=2,
            )

            cv2.putText(
                out,
                key_cell_id,
                (end_x, end_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )

            if start_x is not None and start_y is not None:
                out = cv2.line(
                    out,
                    (end_x, end_y),
                    (start_x, start_y),
                    color=(
                        255,
                        0,
                    ),
                    thickness=2,
                )
            start_x = end_x
            start_y = end_y

        for value_cel_id in kv_item.value:
            value_cell = table.cells.get(value_cel_id)

            end_x = (value_cell.box[0] + value_cell.box[2]) // 2
            end_y = (value_cell.box[1] + value_cell.box[3]) // 2

            cv2.rectangle(
                out,
                (value_cell.box[0], value_cell.box[1]),
                (value_cell.box[2], value_cell.box[3]),
                color=(255, 0, 0),
                thickness=2,
            )

            cv2.putText(
                out,
                value_cell.id,
                (end_x, end_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )

            out = cv2.arrowedLine(
                out,
                (start_x, start_y),
                (end_x, end_y),
                color=(0, 0, 255),
                thickness=2,
                tipLength=0.2,
            )

            start_x = end_x
            start_y = end_y

    return out


def grids_visualizer(
    img, table, value_color=(255, 0, 0), col_color=(0, 255, 0), row_color=(0, 128, 255)
):
    out = img.copy()
    for grid in table.grids:
        for row in grid.rows:
            for cell in row.cells:
                for col in cell.col_keys:
                    col = table.cells.get(col)
                    out = cv2.rectangle(
                        out,
                        (col.box[0], col.box[1]),
                        (col.box[2], col.box[3]),
                        color=col_color,
                        thickness=2,
                    )

                    cx = (col.box[0] + col.box[2]) // 2
                    cy = (col.box[1] + col.box[3]) // 2

                    cv2.putText(
                        out,
                        col.id,
                        (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2,
                    )

                for row_key in cell.row_keys:
                    row_key = table.cells.get(row_key)
                    out = cv2.rectangle(
                        out,
                        (row_key.box[0], row_key.box[1]),
                        (row_key.box[2], row_key.box[3]),
                        color=row_color,
                        thickness=2,
                    )

                    cx = (row_key.box[0] + row_key.box[2]) // 2
                    cy = (row_key.box[1] + row_key.box[3]) // 2

                    cv2.putText(
                        out,
                        row_key.id,
                        (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2,
                    )

                value = table.cells.get(cell.value)

                out = cv2.rectangle(
                    out,
                    (value.box[0], value.box[1]),
                    (value.box[2], value.box[3]),
                    color=value_color,
                    thickness=2,
                )

                cx = (value.box[0] + value.box[2]) // 2
                cy = (value.box[1] + value.box[3]) // 2

                cv2.putText(
                    out,
                    value.id,
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )

    return out


def rec_visualizer(
    img,
    outputs,
    font_path,
    font_size=12,
    font_color=(255, 0, 0),
):
    out = img.copy()
    pillow_img = Image.fromarray(out)
    draw = ImageDraw.Draw(pillow_img)
    has_raqm = features.check_feature(feature="raqm")
    if not has_raqm:
        logger.warning(
            "libraqm is not installed. Vertical text rendering is not supported. Rendering horizontally instead."
        )

    for pred, quad, direction in zip(
        outputs.contents, outputs.points, outputs.directions
    ):
        quad = np.array(quad).astype(np.int32)
        font = ImageFont.truetype(font_path, font_size)
        if direction == "horizontal" or not has_raqm:
            x_offset = 0
            y_offset = -font_size

            pos_x = quad[0][0] + x_offset
            pox_y = quad[0][1] + y_offset
            draw.text((pos_x, pox_y), pred, font=font, fill=font_color)
        else:
            x_offset = -font_size
            y_offset = 0

            pos_x = quad[0][0] + x_offset
            pox_y = quad[0][1] + y_offset
            draw.text(
                (pos_x, pox_y),
                pred,
                font=font,
                fill=font_color,
                direction="ttb",
            )

    out = np.array(pillow_img)
    return out
