import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from ..data.functions import load_image
from ..utils.misc import load_config
from ..utils.visualizer import layout_visualizer
from . import BaseModule
from .models import RTDETR, RTDETRPostProcessor


class LayoutParser(BaseModule):
    def __init__(self, cfg, device="cpu", visualize=False):
        super().__init__(cfg)
        self.cfg = cfg
        self.model = RTDETR(cfg)
        self._device = device
        self.visualize = visualize

        if self.cfg.WEIGHTS:
            self.model.load_state_dict(
                torch.load(self.cfg.WEIGHTS, map_location=self._device)[
                    "model"
                ]
            )

        self.model.eval()
        self.model.to(self._device)

        self.postprocessor = RTDETRPostProcessor(
            num_classes=3,
            num_top_queries=300,
        )

        self.transforms = T.Compose(
            [
                T.Resize((640, 640)),
                T.ToTensor(),
            ]
        )

        self.confidence_threshold = 0.5

    def preprocess(self, img):
        cv_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv_img)
        img_tensor = self.transforms(img)[None].to(self._device)
        return img_tensor

    def postprocess(self, preds, image_size):
        h, w = image_size
        orig_size = torch.tensor([w, h])[None].to(self._device)
        outputs = self.postprocessor(preds, orig_size)

        preds = outputs[0]
        scores = preds["scores"]
        boxes = preds["boxes"][scores > self.confidence_threshold]
        labels = preds["labels"][scores > self.confidence_threshold]
        scores = scores[scores > self.confidence_threshold]

        return {
            "boxes": boxes,
            "labels": labels,
            "scores": scores,
        }

    def __call__(self, img):
        ori_h, ori_w = img.shape[:2]
        img_tensor = self.preprocess(img)

        with torch.inference_mode():
            preds = self.model(img_tensor)
        outputs = self.postprocess(preds, (ori_h, ori_w))

        vis = None
        if self.visualize:
            vis = layout_visualizer(
                outputs,
                img,
            )

        return outputs, vis


cfg = "configs/layout.yaml"
cfg = load_config(cfg)
layout_parser = LayoutParser(cfg.LayoutParser, visualize=True)
img = "dataset/test_20241013/00001256_4521283_7.jpg"
img = load_image(img)
# img = Image.open(img)

outputs, vis = layout_parser(img)
cv2.imwrite("test.jpg", vis)