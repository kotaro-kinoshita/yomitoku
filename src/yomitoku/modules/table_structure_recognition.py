import cv2
import torch
import torchvision.transforms as T
from PIL import Image

from ..data.functions import load_image
from ..utils.misc import load_config
from ..utils.visualizer import layout_visualizer
from . import BaseModule
from .models import RTDETR, RTDETRPostProcessor


class TableStructureRecognition(BaseModule):
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
            num_classes=cfg.RTDETRTransformerv2.num_classes,
            num_top_queries=cfg.RTDETRTransformerv2.num_queries,
        )

        self.transforms = T.Compose(
            [
                T.Resize(cfg.RTDETRTransformerv2.eval_spatial_size),
                T.ToTensor(),
            ]
        )

        self.thresh_score = cfg.thresh_score

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
        boxes = preds["boxes"][scores > self.thresh_score]
        labels = preds["labels"][scores > self.thresh_score]
        scores = scores[scores > self.thresh_score]

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


cfg = "configs/tsr.yaml"
cfg = load_config(cfg)
layout_parser = TableStructureRecognition(
    cfg.TableStructureRecognition, visualize=True
)
img = "dataset/val_20241014_better_table/0000/00001342_6845526_12_0.jpg"
img = load_image(img)

outputs, vis = layout_parser(img)
cv2.imwrite("test.jpg", vis)
