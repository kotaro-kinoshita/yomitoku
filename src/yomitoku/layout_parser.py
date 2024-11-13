from typing import List

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from pydantic import BaseModel, conlist

from .base import BaseModelCatalog, BaseModule
from .configs import LayoutParserRTDETRv2Config
from .models import RTDETRv2
from .postprocessor import RTDETRPostProcessor
from .utils.misc import is_contained
from .utils.visualizer import layout_visualizer


class Element(BaseModel):
    box: conlist(int, min_length=4, max_length=4)
    score: float


class LayoutParserSchema(BaseModel):
    paragraphs: List[Element]
    tables: List[Element]
    figures: List[Element]


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
        elements = self.filtering_elements(outputs[0])
        return elements

    def filtering_elements(self, preds):
        """以下の条件で予測結果のフィルタリングを行う。
        1. 信頼度が閾値以上のものを抽出
        2. 同じクラスに属する矩形のうち、他の矩形の内側に含まれるものを除外
        3. テーブルの内側に存在する、テキストの矩形を除外

        Args:
            outputs (list): _description_

        Returns:
            _type_: _description_
        """

        # 1. 信頼度が閾値以上のものを抽出
        scores = preds["scores"]
        boxes = preds["boxes"][scores > self.thresh_score]
        labels = preds["labels"][scores > self.thresh_score]
        scores = scores[scores > self.thresh_score]

        mapper = {
            id: category for id, category in enumerate(self._cfg.category)
        }
        category_elements = {
            category: {
                "boxes": [],
                "scores": [],
            }
            for category in self._cfg.category
        }

        for box, score, label in zip(boxes, scores, labels):
            category_elements[mapper[label.item()]]["boxes"].append(box)
            category_elements[mapper[label.item()]]["scores"].append(score)

        # 2. 同じクラスに属する矩形のうち、他の矩形の内側に含まれるものを除外
        for category, elements in category_elements.items():
            group_box = elements["boxes"]
            group_score = elements["scores"]
            check_list = [True] * len(group_box)
            for i, box_i in enumerate(group_box):
                for j, box_j in enumerate(group_box):
                    if i >= j:
                        continue

                    ij = is_contained(box_i, box_j)
                    ji = is_contained(box_j, box_i)

                    box_i_area = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
                    box_j_area = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1])

                    if ij and ji:
                        if box_i_area > box_j_area:
                            check_list[j] = False
                        else:
                            check_list[i] = False
                    elif ij:
                        check_list[j] = False
                    elif ji:
                        check_list[i] = False

            category_elements[category]["boxes"] = np.array(group_box)[
                check_list
            ]
            category_elements[category]["scores"] = np.array(group_score)[
                check_list
            ]

        # 3. テーブルの内側に存在する、テキスト矩形を除外
        paragraph_boxes = category_elements["paragraphs"]["boxes"]
        table_boxes = category_elements["tables"]["boxes"]

        check_list = [True] * len(paragraph_boxes)
        for i, table_box in enumerate(table_boxes):
            for j, paragraph_box in enumerate(paragraph_boxes):
                if is_contained(table_box, paragraph_box):
                    check_list[j] = False

        category_elements["paragraphs"]["boxes"] = paragraph_boxes[check_list]
        category_elements["paragraphs"]["scores"] = category_elements[
            "paragraphs"
        ]["scores"][check_list]

        outputs = {category: [] for category in self._cfg.category}

        for category, elements in category_elements.items():
            outputs[category] = [
                {
                    "box": box.astype(int).tolist(),
                    "score": float(score),
                }
                for box, score in zip(elements["boxes"], elements["scores"])
            ]

        return outputs

    def __call__(self, img):
        ori_h, ori_w = img.shape[:2]
        img_tensor = self.preprocess(img)

        with torch.inference_mode():
            preds = self.model(img_tensor)
        outputs = self.postprocess(preds, (ori_h, ori_w))
        results = LayoutParserSchema(**outputs)

        vis = None
        if self.visualize:
            vis = layout_visualizer(
                results,
                img,
            )

        return results, vis
