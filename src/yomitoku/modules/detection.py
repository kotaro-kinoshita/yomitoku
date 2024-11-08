import torch

import numpy as np

from . import BaseModule
from .models import DBNet

from ..data.functions import (
    resize_shortest_edge,
    array_to_tensor,
    normalize_image,
)

from .representer import SegDetectorRepresenter
from ..utils.visualizer import det_visualizer


class Detection(BaseModule):
    def __init__(self, cfg, device="cpu", visualize=False):
        super().__init__(cfg)
        self.cfg = cfg.DETECTION
        self.model = DBNet(self.cfg)

        self._device = device
        self.visualize = visualize

        if self.cfg.WEIGHTS:
            self.model.load_state_dict(
                torch.load(self.cfg.WEIGHTS, map_location=self._device)["model"]
            )

        self.model.eval()
        self.model.to(self._device)

        self.post_processor = SegDetectorRepresenter(self.cfg.POST_PROCESS)

    def preprocess(self, img):
        img = img.copy()
        img = img[:, :, ::-1].astype(np.float32)
        resized = resize_shortest_edge(
            img, self.cfg.DATA.SHORTEST_SIZE, self.cfg.DATA.LIMIT_SIZE
        )
        normalized = normalize_image(resized)
        tensor = array_to_tensor(normalized)
        return tensor

    def postprocess(self, preds, image_size):
        return self.post_processor(preds, image_size)

    def __call__(self, img):
        """apply the detection model to the input image.

        Args:
            img (np.ndarray): target image(BGR)

        Returns:
            list: list of quadrilaterals. Each quadrilateral is represented as a list of 4 points sorted clockwise.
            list: list of confidence scores
            np.ndarray: rendering image
        """

        ori_h, ori_w = img.shape[:2]
        tensor = self.preprocess(img)
        tensor = tensor.to(self._device)
        with torch.inference_mode():
            preds = self.model(tensor)

        quads, scores = self.postprocess(preds, (ori_h, ori_w))

        vis = None
        if self.visualize:
            vis = det_visualizer(
                preds,
                img,
                quads,
                vis_heatmap=self.cfg.VISUALIZE.HEATMAP,
                line_color=tuple(self.cfg.VISUALIZE.COLOR[::-1]),
            )

        return quads, scores, vis
