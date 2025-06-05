import os

from PIL import Image
from io import BytesIO

from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import stringWidth

import numpy as np
import jaconv

from ..constants import ROOT_DIR

FONT_PATH = ROOT_DIR + "/resource/MPLUS1p-Medium.ttf"


def _poly2rect(points):
    """
    Convert a polygon defined by its corner points to a rectangle.
    The points should be in the format [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
    """
    points = np.array(points, dtype=int)
    x_min = points[:, 0].min()
    x_max = points[:, 0].max()
    y_min = points[:, 1].min()
    y_max = points[:, 1].max()

    return [x_min, y_min, x_max, y_max]


def _calc_font_size(content, bbox_height, bbox_width):
    rates = np.arange(0.5, 1.0, 0.01)

    min_diff = np.inf
    best_font_size = None
    for rate in rates:
        font_size = bbox_height * rate
        text_w = stringWidth(content, "MPLUS1p-Medium", font_size)
        diff = abs(text_w - bbox_width)
        if diff < min_diff:
            min_diff = diff
            best_font_size = font_size

    return best_font_size


def to_full_width(text):
    fw_map = {
        "\u00a5": "\uffe5",  # ¥ → ￥
        "\u00b7": "\u30fb",  # · → ・
        " ": "\u3000",  # 半角スペース→全角スペース
    }

    TO_FULLWIDTH = str.maketrans(fw_map)

    jaconv_text = jaconv.h2z(text, kana=True, ascii=True, digit=True)
    jaconv_text = jaconv_text.translate(TO_FULLWIDTH)

    return jaconv_text


def create_searchable_pdf(images, ocr_results, output_path, font_path=None):
    if font_path is None:
        font_path = FONT_PATH

    pdfmetrics.registerFont(TTFont("MPLUS1p-Medium", font_path))

    packet = BytesIO()
    c = canvas.Canvas(packet)

    for i, (image, ocr_result) in enumerate(zip(images, ocr_results)):
        image = Image.fromarray(image[:, :, ::-1])  # Convert BGR to RGB
        pdfmetrics.registerFont(TTFont("MPLUS1p-Medium", FONT_PATH))

        image_path = f"tmp_{i}.png"
        image.save(image_path)
        w, h = image.size

        c.setPageSize((w, h))
        c.drawImage(image_path, 0, 0, width=w, height=h)
        os.remove(image_path)  # Clean up temporary image file

        for word in ocr_result.words:
            text = word.content
            bbox = _poly2rect(word.points)
            direction = word.direction

            x1, y1, x2, y2 = bbox
            bbox_height = y2 - y1
            bbox_width = x2 - x1

            if direction == "vertical":
                text = to_full_width(text)

            if direction == "horizontal":
                font_size = _calc_font_size(text, bbox_height, bbox_width)
            else:
                font_size = _calc_font_size(text, bbox_width, bbox_height)

            c.setFont("MPLUS1p-Medium", font_size)
            c.setFillColorRGB(1, 1, 1, alpha=0)  # 透明
            if direction == "vertical":
                base_y = h - y2 + (bbox_height - font_size)
                for j, ch in enumerate(text):
                    c.saveState()
                    c.translate(x1 + font_size * 0.5, base_y - (j - 1) * font_size)
                    c.rotate(-90)
                    c.drawString(0, 0, ch)
                    c.restoreState()
            else:
                base_y = h - y2 + (bbox_height - font_size) * 0.5
                c.drawString(x1, base_y, text)
        c.showPage()
    c.save()

    with open(output_path, "wb") as f:
        f.write(packet.getvalue())
