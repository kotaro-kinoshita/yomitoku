import numpy as np
import torch

from .base import BaseModule
from .data.functions import (
    array_to_tensor,
    normalize_image,
    resize_shortest_edge,
)
from .models import DBNetPlus
from .postprocessor import DBnetPostProcessor
from .utils.visualizer import det_visualizer


class TextDetector(BaseModule):
    def __init__(self, path_cfg=None, device="cpu", visualize=False):
        super().__init__()
        self.set_config(path_cfg)
        self.model = DBNetPlus.from_pretrained(
            self._cfg.hf_hub_repo, cfg=self._cfg
        )

        self._device = device
        self.visualize = visualize

        self.model.eval()
        self.model.to(self._device)

        self.post_processor = DBnetPostProcessor(**self._cfg.post_process)

    def preprocess(self, img):
        img = img.copy()
        img = img[:, :, ::-1].astype(np.float32)
        resized = resize_shortest_edge(
            img, self._cfg.data.shortest_size, self._cfg.data.limit_size
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
        outputs = {"quads": quads, "scores": scores}

        vis = None
        if self.visualize:
            vis = det_visualizer(
                preds,
                img,
                quads,
                vis_heatmap=self._cfg.visualize.heatmap,
                line_color=tuple(self._cfg.visualize.color[::-1]),
            )

        return outputs, vis
