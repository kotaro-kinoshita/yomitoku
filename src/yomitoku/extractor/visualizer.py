import os
from typing import List

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..constants import ROOT_DIR
from .resolver import ResolvedField

_CONFIDENCE_ALPHA = {
    "high": 0.15,
    "medium": 0.25,
    "low": 0.4,
}

_DEFAULT_FONT_PATH = os.path.join(ROOT_DIR, "resource", "MPLUS1p-Medium.ttf")
_DEFAULT_FONT_SIZE = 14

# BGR: bright red - high visibility on most document backgrounds
_COLOR_BGR = (0, 0, 255)
_COLOR_PIL = _COLOR_BGR  # array is BGR; PIL writes the same byte values


def extraction_visualizer(
    img: np.ndarray,
    fields: List[ResolvedField],
    font_path: str = _DEFAULT_FONT_PATH,
    font_size: int = _DEFAULT_FONT_SIZE,
) -> np.ndarray:
    out = img.copy()
    overlay = img.copy()

    font = ImageFont.truetype(font_path, font_size)

    for field in fields:
        alpha = _CONFIDENCE_ALPHA.get(field.confidence, 0.2)

        for elem in field.elements:
            x1, y1, x2, y2 = map(int, elem.box)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), _COLOR_BGR, -1)
            cv2.rectangle(out, (x1, y1), (x2, y2), _COLOR_BGR, 2)

            label = elem.label if elem.label else field.name
            pil_img = Image.fromarray(out)
            draw = ImageDraw.Draw(pil_img)

            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

            label_y = max(y1 - th - 4, 0)

            draw.rectangle(
                [(x1, label_y), (x1 + tw + 4, label_y + th + 4)],
                fill=_COLOR_PIL,
            )
            draw.text(
                (x1 + 2, label_y + 2),
                label,
                font=font,
                fill=(255, 255, 255),
            )
            out = np.array(pil_img)

        out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
        overlay = out.copy()

    return out
