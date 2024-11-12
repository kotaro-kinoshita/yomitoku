import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from .base import BaseModelCatalog, BaseModule
from .configs import LayoutParserRTDETRv2Config
from .models import RTDETRv2
from .postprocessor import RTDETRPostProcessor
from .utils.visualizer import layout_visualizer


class LayoutParserModelCatalog(BaseModelCatalog):
    def __init__(self):
        super().__init__()
        self.register("rtdetrv2", LayoutParserRTDETRv2Config, RTDETRv2)


def is_contained(rect_a, rect_b, threshold=0.75):
    """二つの矩形A, Bが与えられたとき、矩形Bが矩形Aに含まれるかどうかを判定する。
    ずれを許容するため、重複率求め、thresholdを超える場合にTrueを返す。


    Args:
        rect_a (np.array): x1, y1, x2, y2
        rect_b (np.array): x1, y1, x2, y2
        threshold (float, optional): 判定の閾値. Defaults to 0.9.

    Returns:
        bool: 矩形Bが矩形Aに含まれる場合True
    """

    ax1, ay1, ax2, ay2 = rect_a
    bx1, by1, bx2, by2 = rect_b

    # 交差領域の左上と右下の座標
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    # 矩形が交差しているかを確認
    if ix1 < ix2 and iy1 < iy2:
        # 交差領域の幅と高さ
        intersection_width = ix2 - ix1
        intersection_height = iy2 - iy1
        intersection_area = intersection_width * intersection_height

        b_area = (bx2 - bx1) * (by2 - by1)

        if intersection_area / b_area > threshold:
            return True

    return ax1 <= bx1 and ay1 <= by1 and ax2 >= bx2 and ay2 >= by2


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

    def filter(self, preds):
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
        paragraph_boxes = category_elements["paragraph"]["boxes"]
        table_boxes = category_elements["table"]["boxes"]

        check_list = [True] * len(paragraph_boxes)
        for i, table_box in enumerate(table_boxes):
            for j, paragraph_box in enumerate(paragraph_boxes):
                if is_contained(table_box, paragraph_box):
                    check_list[j] = False

        category_elements["paragraph"]["boxes"] = np.array(paragraph_boxes)[
            check_list
        ]
        category_elements["paragraph"]["scores"] = np.array(
            category_elements["paragraph"]["scores"]
        )[check_list]

        return category_elements

    def postprocess(self, preds, image_size):
        h, w = image_size
        orig_size = torch.tensor([w, h])[None].to(self.device)
        outputs = self.postprocessor(preds, orig_size)
        elements = self.filter(outputs[0])

        return elements

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
