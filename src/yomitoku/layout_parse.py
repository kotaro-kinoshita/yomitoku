import cv2
import torch
import torchvision.transforms as T
from omegaconf import OmegaConf
from PIL import Image

from .base import BaseModule
from .configs import LayoutParserConfig
from .data.functions import load_image
from .models import RTDETR
from .postprocessor import RTDETRPostProcessor
from .utils.misc import load_config
from .utils.visualizer import layout_visualizer


class LayoutParser(BaseModule):
    def __init__(self, path_cfg=None, device="cpu", visualize=False):
        super().__init__()
        self.cfg = self.set_config(path_cfg)
        self.model = RTDETR.from_pretrained(self.cfg.hf_hub_repo, cfg=self.cfg)
        self._device = device
        self.visualize = visualize

        self.model.eval()
        self.model.to(self._device)

        self.postprocessor = RTDETRPostProcessor(
            num_classes=self.cfg.RTDETRTransformerv2.num_classes,
            num_top_queries=self.cfg.RTDETRTransformerv2.num_queries,
        )

        self.transforms = T.Compose(
            [
                T.Resize(self.cfg.data.img_size),
                T.ToTensor(),
            ]
        )

        self.thresh_score = self.cfg.thresh_score

    def set_config(self, path_cfg):
        cfg = OmegaConf.structured(LayoutParserConfig)
        if path_cfg is not None:
            yaml_config = load_config(path_cfg)
            cfg = OmegaConf.merge(cfg, yaml_config)
        return cfg

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


cfg = "configs/layout_parse.yaml"
layout_parser = LayoutParser(path_cfg=None, visualize=True)
img = "dataset/test_20241013/00001256_4521283_7.jpg"
img = load_image(img)
# img = Image.open(img)

outputs, vis = layout_parser(img)
cv2.imwrite("test.jpg", vis)
