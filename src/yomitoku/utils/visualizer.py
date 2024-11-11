import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PALETTE = [
    [0, 192, 64],
    [0, 64, 96],
    [128, 192, 192],
    [0, 64, 64],
    [0, 192, 224],
    [0, 192, 192],
    [128, 192, 64],
    [0, 192, 96],
    [128, 192, 64],
    [128, 32, 192],
    [0, 0, 224],
    [0, 0, 64],
    [0, 160, 192],
    [128, 0, 96],
    [128, 0, 192],
    [0, 32, 192],
    [128, 128, 224],
    [0, 0, 192],
    [128, 160, 192],
    [128, 128, 0],
    [128, 0, 32],
    [128, 32, 0],
    [128, 0, 128],
    [64, 128, 32],
    [0, 160, 0],
    [0, 0, 0],
    [192, 128, 160],
    [0, 32, 0],
    [0, 128, 128],
    [64, 128, 160],
    [128, 160, 0],
    [0, 128, 0],
    [192, 128, 32],
    [128, 96, 128],
    [0, 0, 128],
    [64, 0, 32],
    [0, 224, 128],
    [128, 0, 0],
    [192, 0, 160],
    [0, 96, 128],
    [128, 128, 128],
    [64, 0, 160],
    [128, 224, 128],
    [128, 128, 64],
    [192, 0, 32],
    [128, 96, 0],
    [128, 0, 192],
    [0, 128, 32],
    [64, 224, 0],
    [0, 0, 64],
    [128, 128, 160],
    [64, 96, 0],
    [0, 128, 192],
    [0, 128, 160],
    [192, 224, 0],
    [0, 128, 64],
    [128, 128, 32],
    [192, 32, 128],
    [0, 64, 192],
    [0, 0, 32],
    [64, 160, 128],
    [128, 64, 64],
    [128, 0, 160],
    [64, 32, 128],
    [128, 192, 192],
    [0, 0, 160],
    [192, 160, 128],
    [128, 192, 0],
    [128, 0, 96],
    [192, 32, 0],
    [128, 64, 128],
    [64, 128, 96],
    [64, 160, 0],
    [0, 64, 0],
    [192, 128, 224],
    [64, 32, 0],
    [0, 192, 128],
    [64, 128, 224],
    [192, 160, 0],
    [0, 192, 0],
    [192, 128, 96],
    [192, 96, 128],
    [0, 64, 128],
    [64, 0, 96],
    [64, 224, 128],
    [128, 64, 0],
    [192, 0, 224],
    [64, 96, 128],
    [128, 192, 128],
    [64, 0, 224],
    [192, 224, 128],
    [128, 192, 64],
    [192, 0, 96],
    [192, 96, 0],
    [128, 64, 192],
    [0, 128, 96],
    [0, 224, 0],
    [64, 64, 64],
    [128, 128, 224],
    [0, 96, 0],
    [64, 192, 192],
    [0, 128, 224],
    [128, 224, 0],
    [64, 192, 64],
    [128, 128, 96],
    [128, 32, 128],
    [64, 0, 192],
    [0, 64, 96],
    [0, 160, 128],
    [192, 0, 64],
    [128, 64, 224],
    [0, 32, 128],
    [192, 128, 192],
    [0, 64, 224],
    [128, 160, 128],
    [192, 128, 0],
    [128, 64, 32],
    [128, 32, 64],
    [192, 0, 128],
    [64, 192, 32],
    [0, 160, 64],
    [64, 0, 0],
    [192, 192, 160],
    [0, 32, 64],
    [64, 128, 128],
    [64, 192, 160],
    [128, 160, 64],
    [64, 128, 0],
    [192, 192, 32],
    [128, 96, 192],
    [64, 0, 128],
    [64, 64, 32],
    [0, 224, 192],
    [192, 0, 0],
    [192, 64, 160],
    [0, 96, 192],
    [192, 128, 128],
    [64, 64, 160],
    [128, 224, 192],
    [192, 128, 64],
    [192, 64, 32],
    [128, 96, 64],
    [192, 0, 192],
    [0, 192, 32],
    [64, 224, 64],
    [64, 0, 64],
    [128, 192, 160],
    [64, 96, 64],
    [64, 128, 192],
    [0, 192, 160],
    [192, 224, 64],
    [64, 128, 64],
    [128, 192, 32],
    [192, 32, 192],
    [64, 64, 192],
    [0, 64, 32],
    [64, 160, 192],
    [192, 64, 64],
    [128, 64, 160],
    [64, 32, 192],
    [192, 192, 192],
    [0, 64, 160],
    [192, 160, 192],
    [192, 192, 0],
    [128, 64, 96],
    [192, 32, 64],
    [192, 64, 128],
    [64, 192, 96],
    [64, 160, 64],
    [64, 64, 0],
]


def det_visualizer(
    preds, img, quads, vis_heatmap=False, line_color=(0, 255, 0)
):
    preds = preds["thresh_binary"][0]
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


def layout_visualizer(preds, img):
    classes = preds["labels"].cpu().numpy()
    boxes = preds["boxes"].cpu().numpy()
    out = img.copy()

    for (
        box,
        category,
    ) in zip(boxes, classes):
        color = PALETTE[category % len(PALETTE)]
        box = box.astype(np.int32)
        out = cv2.rectangle(out, (box[0], box[1]), (box[2], box[3]), color, 1)

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

    for pred, quad in zip(outputs["contents"], outputs["quads"]):
        quad = np.array(quad).astype(np.int32)
        word_width = np.linalg.norm(quad[0] - quad[1])
        word_height = np.linalg.norm(quad[1] - quad[2])
        font = ImageFont.truetype(font_path, font_size)

        direction = "horizontal"
        if word_width * 2 < word_height:
            direction = "vertical"

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
