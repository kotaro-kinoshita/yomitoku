import cv2
import torch
import torchvision.transforms as T
from PIL import Image

from .base import BaseModelCatalog, BaseModule
from .configs import LayoutParserRTDETRv2Config
from .data.functions import load_image
from .models import RTDETRv2
from .postprocessor import RTDETRPostProcessor
from .utils.visualizer import layout_visualizer


class LayoutParserModelCatalog(BaseModelCatalog):
    def __init__(self):
        super().__init__()
        self.register("rtdetrv2", LayoutParserRTDETRv2Config, RTDETRv2)


class LayoutParser(BaseModule):
    model_catalog = LayoutParserModelCatalog()

    def __init__(
        self,
        model_name="rtdetrv2",
        path_cfg=None,
        device="cuda",
        visualize=False,
    ):
        super().__init__()
        self.load_model(model_name, path_cfg)
        self.device = device
        self.visualize = visualize

        self.model.eval()
        self.model.to(self.device)

        self.postprocessor = RTDETRPostProcessor(
            num_classes=self._cfg.RTDETRTransformerv2.num_classes,
            num_top_queries=self._cfg.RTDETRTransformerv2.num_queries,
        )

        self.transforms = T.Compose(
            [
                T.Resize(self._cfg.data.img_size),
                T.ToTensor(),
            ]
        )

        self.thresh_score = self._cfg.thresh_score

    def preprocess(self, img):
        cv_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv_img)
        img_tensor = self.transforms(img)[None].to(self.device)
        return img_tensor

    def postprocess(self, preds, image_size):
        h, w = image_size
        orig_size = torch.tensor([w, h])[None].to(self.device)
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


LayoutParser.catalog()

cfg = "configs/layout_parse.yaml"
layout_parser = LayoutParser(path_cfg=None, visualize=True)
img = "dataset/test_20241013/00001256_4521283_7.jpg"
img = load_image(img)
# img = Image.open(img)

layout_parser.log_config()
layout_parser.save_config("test.yaml")

outputs, vis = layout_parser(img)
cv2.imwrite("test.jpg", vis)