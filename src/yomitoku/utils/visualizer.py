import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, features
from ..constants import PALETTE
from .logger import set_logger

logger = set_logger(__name__, "INFO")


def _reading_order_visualizer(img, elements, line_color, tip_size):
    out = img.copy()
    for i, element in enumerate(elements):
        if i == 0:
            continue

        prev_element = elements[i - 1]
        cur_x1, cur_y1, cur_x2, cur_y2 = element.box
        prev_x1, prev_y1, prev_x2, prev_y2 = prev_element.box

        cur_center = (
            cur_x1 + (cur_x2 - cur_x1) / 2,
            cur_y1 + (cur_y2 - cur_y1) / 2,
        )
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


def det_visualizer(img, quads, pred, vis_heatmap=False, line_color=(0, 255, 0)):
    out = img.copy()
    h, w = out.shape[:2]
    if vis_heatmap:
        preds = pred
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
    results_dict = results.dict()
    for id, (category, preds) in enumerate(results_dict.items()):
        for element in preds:
            box = element["box"]
            role = element["role"]

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
