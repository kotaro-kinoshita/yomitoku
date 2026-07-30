import cv2
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms as T

from .functions import (
    calc_resize_without_padding,
    extract_roi_with_perspective,
    resize_with_dynamic_padding,
    resize_with_padding,
    rotate_text_image,
    validate_quads,
)

from concurrent.futures import ThreadPoolExecutor


def _quad_short_sides(quads):
    q = np.asarray(quads, dtype=np.float32).reshape(-1, 4, 2)
    w = np.linalg.norm(q[:, 0] - q[:, 1], axis=1)
    h = np.linalg.norm(q[:, 1] - q[:, 2], axis=1)
    return np.maximum(1.0, np.minimum(w, h))


def _calc_source_levels(quads, target_height, max_level=3):
    """Per-quad pyramid level for recognition cropping.

    The recognizer resizes every crop so its short side is at most
    `target_height` px (it never upscales), so a crop whose short side is
    `s` can be cut from a 2^k-downscaled copy of the image as long as
    s / 2^k >= target_height — the recognizer sees the same effective
    resolution. Large text therefore crops from a small pyramid level
    (perspective-warp cost drops ~4^k) while small text keeps full
    resolution. Returns an int level per quad (0 = original image).
    """
    if len(quads) == 0:
        return np.zeros(0, dtype=int)
    short = _quad_short_sides(quads)
    levels = np.floor(np.log2(short / float(target_height))).astype(int)
    return np.clip(levels, 0, max_level)


class ParseqDataset(Dataset):
    def __init__(
        self,
        cfg,
        img,
        quads,
        num_workers=8,
        dynamic_width=False,
        source_downscale=False,
    ):
        self.quads = quads
        self.cfg = cfg
        self.dynamic_width = dynamic_width
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(0.5, 0.5),
            ]
        )

        # Crops are downsampled to a 32px-tall canvas anyway, so large text
        # can be cut from a 2^k-downscaled copy of the image without losing
        # information the recognizer uses. Build only the needed pyramid
        # levels (exact 2x INTER_AREA halvings) and route each quad to its
        # level; small text keeps cropping from the original image.
        self._levels = {0: img[:, :, ::-1]}
        quad_levels = np.zeros(len(quads), dtype=int)
        if source_downscale and len(quads) > 0:
            quad_levels = _calc_source_levels(quads, cfg.data.img_size[0])
            level_img = img
            for k in range(1, int(quad_levels.max()) + 1):
                level_img = cv2.resize(
                    level_img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
                )
                if (quad_levels >= k).any():
                    self._levels[k] = level_img[:, :, ::-1]

        self.img = self._levels[0]

        jobs = [
            (quad, int(level), self._levels.get(int(level), self.img))
            for quad, level in zip(quads, quad_levels)
        ]

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            data = list(executor.map(self._preprocess_job, jobs))

        self.data = [d[0] for d in data if d is not None]
        self.roi_images = [d[1] for d in data if d is not None]
        self.content_widths = [d[2] for d in data if d is not None]
        # Reported coordinates stay in the ORIGINAL image space.
        self.valid_quads = [q for q, d in zip(self.quads, data) if d is not None]

    def _preprocess_job(self, job):
        quad, level, level_img = job
        if level > 0:
            quad = (np.asarray(quad, dtype=np.float32) / (2.0**level)).tolist()
        return self._preprocess_on(level_img, quad)

    def preprocess(self, quad):
        return self._preprocess_on(self.img, quad)

    def _preprocess_on(self, img, quad):
        if validate_quads(img, quad) is None:
            return None

        roi_img = extract_roi_with_perspective(img, quad)

        if roi_img is None:
            return None

        roi_img = rotate_text_image(roi_img, thresh_aspect=2)
        if self.dynamic_width:
            resized = resize_with_dynamic_padding(roi_img, self.cfg.data.img_size)
        else:
            resized = resize_with_padding(roi_img, self.cfg.data.img_size)

        _, content_width = calc_resize_without_padding(roi_img, self.cfg.data.img_size)

        return resized, roi_img, content_width

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.transform(self.data[index])
