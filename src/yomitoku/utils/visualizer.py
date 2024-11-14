import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PALETTE = [
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [0, 255, 255],
    [255, 0, 255],
    [128, 0, 0],
    [0, 128, 0],
    [0, 0, 128],
    [255, 128, 0],
    [0, 255, 128],
    [128, 0, 255],
    [128, 255, 0],
    [0, 128, 255],
    [255, 0, 128],
    [255, 128, 128],
    [128, 255, 128],
    [128, 128, 255],
    [255, 255, 128],
    [255, 128, 255],
    [128, 255, 255],
    [128, 128, 128],
]


def det_visualizer(preds, img, quads, vis_heatmap=False, line_color=(0, 255, 0)):
    preds = preds["binary"][0]
    binary = preds.detach().cpu().numpy()
    out = img.copy()
    h, w = out.shape[:2]
    binary = binary.squeeze(0)
    binary = (binary * 255).astype(np.uint8)
    if vis_heatmap:
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
            color = PALETTE[id % len(PALETTE)]
            x1, y1, x2, y2 = tuple(map(int, box))
            out = cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            out = cv2.putText(
                out,
                category,
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

        text = f"({row}, {col}) {row_span}x{col_span}"

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
    font_size=12,
    font_color=(255, 0, 0),
    font_path="resource/MPLUS1p-Medium.ttf",
):
    out = img.copy()
    pillow_img = Image.fromarray(out)
    draw = ImageDraw.Draw(pillow_img)

    for pred, quad, direction in zip(
        outputs.contents, outputs.points, outputs.directions
    ):
        quad = np.array(quad).astype(np.int32)
        font = ImageFont.truetype(font_path, font_size)
        if direction == "horizontal":
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
