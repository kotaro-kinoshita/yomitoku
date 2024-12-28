from typing import List, Union

import cv2
import os
import onnx
import onnxruntime
import torch
import torchvision.transforms as T
from PIL import Image
from pydantic import conlist

from .constants import ROOT_DIR

from .base import BaseModelCatalog, BaseModule, BaseSchema
from .configs import LayoutParserRTDETRv2Config
from .models import RTDETRv2
from .postprocessor import RTDETRPostProcessor
from .utils.misc import filter_by_flag, is_contained
from .utils.visualizer import layout_visualizer


class Element(BaseSchema):
    box: conlist(int, min_length=4, max_length=4)
    score: float
    role: Union[str, None]


class LayoutParserSchema(BaseSchema):
    paragraphs: List[Element]
    tables: List[Element]
    figures: List[Element]


class LayoutParserModelCatalog(BaseModelCatalog):
    def __init__(self):
        super().__init__()
        self.register("rtdetrv2", LayoutParserRTDETRv2Config, RTDETRv2)


def filter_contained_rectangles_within_category(category_elements):
    """同一カテゴリに属する矩形のうち、他の矩形の内側に含まれるものを除外"""

    for category, elements in category_elements.items():
        group_box = [element["box"] for element in elements]
        check_list = [True] * len(group_box)
        for i, box_i in enumerate(group_box):
            for j, box_j in enumerate(group_box):
                if i >= j:
                    continue

                ij = is_contained(box_i, box_j)
                ji = is_contained(box_j, box_i)

                box_i_area = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
                box_j_area = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1])

                # 双方から見て内包関係にある場合、面積の大きい方を残す
                if ij and ji:
                    if box_i_area > box_j_area:
                        check_list[j] = False
                    else:
                        check_list[i] = False
                elif ij:
                    check_list[j] = False
                elif ji:
                    check_list[i] = False

        category_elements[category] = filter_by_flag(elements, check_list)

    return category_elements


def filter_contained_rectangles_across_categories(category_elements, source, target):
    """sourceカテゴリの矩形がtargetカテゴリの矩形に内包される場合、sourceカテゴリの矩形を除外"""

    src_boxes = [element["box"] for element in category_elements[source]]
    tgt_boxes = [element["box"] for element in category_elements[target]]

    check_list = [True] * len(tgt_boxes)
    for i, src_box in enumerate(src_boxes):
        for j, tgt_box in enumerate(tgt_boxes):
            if is_contained(src_box, tgt_box):
                check_list[j] = False

    category_elements[target] = filter_by_flag(category_elements[target], check_list)
    return category_elements


class LayoutParser(BaseModule):
    model_catalog = LayoutParserModelCatalog()

    def __init__(
        self,
        model_name="rtdetrv2",
        path_cfg=None,
        device="cuda",
        visualize=False,
        from_pretrained=True,
        infer_onnx=False,
    ):
        super().__init__()
        self.load_model(model_name, path_cfg, from_pretrained)
        self.device = device
        self.visualize = visualize

        self.model.eval()

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

        self.label_mapper = {
            id: category for id, category in enumerate(self._cfg.category)
        }

        self.role = self._cfg.role
        self.infer_onnx = infer_onnx
        if infer_onnx:
            name = self._cfg.hf_hub_repo.split("/")[-1]
            path_onnx = f"{ROOT_DIR}/onnx/{name}.onnx"
            if not os.path.exists(path_onnx):
                self.convert_onnx(path_onnx)

            self.model = None

            model = onnx.load(path_onnx)
            if torch.cuda.is_available() and device == "cuda":
                self.sess = onnxruntime.InferenceSession(
                    model.SerializeToString(), providers=["CUDAExecutionProvider"]
                )
            else:
                self.sess = onnxruntime.InferenceSession(model.SerializeToString())

        if self.model is not None:
            self.model.to(self.device)

    def convert_onnx(self, path_onnx):
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

        img_size = self._cfg.data.img_size
        dummy_input = torch.randn(1, 3, *img_size, requires_grad=True)

        torch.onnx.export(
            self.model,
            dummy_input,
            path_onnx,
            opset_version=16,
            input_names=["input"],
            output_names=["pred_logits", "pred_boxes"],
            dynamic_axes=dynamic_axes,
        )

    def preprocess(self, img):
        cv_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv_img)
        img_tensor = self.transforms(img)[None]
        return img_tensor

    def postprocess(self, preds, image_size):
        h, w = image_size
        orig_size = torch.tensor([w, h])[None].to(self.device)
        outputs = self.postprocessor(preds, orig_size, self.thresh_score)
        outputs = self.filtering_elements(outputs[0])
        results = LayoutParserSchema(**outputs)
        return results

    def filtering_elements(self, preds):
        scores = preds["scores"]
        boxes = preds["boxes"]
        labels = preds["labels"]

        category_elements = {
            category: []
            for category in self.label_mapper.values()
            if category not in self.role
        }

        for box, score, label in zip(boxes, scores, labels):
            category = self.label_mapper[label.item()]

            role = None
            if category in self.role:
                role = category
                category = "paragraphs"

            category_elements[category].append(
                {
                    "box": box.astype(int).tolist(),
                    "score": float(score),
                    "role": role,
                }
            )

        category_elements = filter_contained_rectangles_within_category(
            category_elements
        )

        category_elements = filter_contained_rectangles_across_categories(
            category_elements, "tables", "paragraphs"
        )

        return category_elements

    def __call__(self, img):
        ori_h, ori_w = img.shape[:2]
        img_tensor = self.preprocess(img)

        if self.infer_onnx:
            input = img_tensor.numpy()
            results = self.sess.run(None, {"input": input})
            preds = {
                "pred_logits": torch.tensor(results[0]).to(self.device),
                "pred_boxes": torch.tensor(results[1]).to(self.device),
            }

        else:
            with torch.inference_mode():
                img_tensor = img_tensor.to(self.device)
                preds = self.model(img_tensor)

        results = self.postprocess(preds, (ori_h, ori_w))

        vis = None
        if self.visualize:
            vis = layout_visualizer(
                results,
                img,
            )

        return results, vis
