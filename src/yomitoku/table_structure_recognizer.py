import cv2
import torch
import torchvision.transforms as T
from PIL import Image

from .base import BaseModelCatalog, BaseModule
from .configs import TableStructureRecognizerRTDETRv2Config
from .models import RTDETRv2
from .postprocessor import RTDETRPostProcessor
from .utils.misc import calc_intersection, is_contained
from .utils.visualizer import table_visualizer


class TableStructureRecognizerModelCatalog(BaseModelCatalog):
    def __init__(self):
        super().__init__()
        self.register(
            "rtdetrv2", TableStructureRecognizerRTDETRv2Config, RTDETRv2
        )


class TableStructureRecognizer(BaseModule):
    model_catalog = TableStructureRecognizerModelCatalog()

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

    def preprocess(self, img, boxes):
        cv_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        table_imgs = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            table_img = cv_img[y1:y2, x1:x2, :]
            th, hw = table_img.shape[:2]
            table_img = Image.fromarray(table_img)
            img_tensor = self.transforms(table_img)[None].to(self.device)
            table_imgs.append(
                {
                    "tensor": img_tensor,
                    "size": (th, hw),
                    "offset": (x1, y1),
                }
            )
        return table_imgs

    def postprocess(self, preds, data):
        h, w = data["size"]
        orig_size = torch.tensor([w, h])[None].to(self.device)
        outputs = self.postprocessor(preds, orig_size)

        # 1. 信頼度が閾値以上のものを抽出
        preds = outputs[0]
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
            box[0] += data["offset"][0]
            box[1] += data["offset"][1]
            box[2] += data["offset"][0]
            box[3] += data["offset"][1]

            category_elements[mapper[label.item()]]["boxes"].append(box)
            category_elements[mapper[label.item()]]["scores"].append(score)

        cells, n_row, n_col = self.calculate_cell(category_elements)

        table_x, table_y = data["offset"]
        table_x2 = table_x + data["size"][1]
        table_y2 = table_y + data["size"][0]

        table = {
            "box": [table_x, table_y, table_x2, table_y2],
            "n_row": n_row,
            "n_col": n_col,
            "cells": cells,
        }

        return table

    def calculate_cell(self, elements):
        row_boxes = elements["row"]["boxes"]
        col_boxes = elements["col"]["boxes"]
        span_boxes = elements["span"]["boxes"]

        row_boxes = sorted(row_boxes, key=lambda x: x[1])
        col_boxes = sorted(col_boxes, key=lambda x: x[0])

        sub_cells = []
        for i, row_box in enumerate(row_boxes):
            for j, col_box in enumerate(col_boxes):
                intersection = calc_intersection(row_box, col_box)
                if intersection is None:
                    continue

                sub_cells.append(
                    {
                        "col": j + 1,
                        "row": i + 1,
                        "col_span": 1,
                        "row_span": 1,
                        "box": intersection,
                    }
                )

        check_list = [True] * len(sub_cells)
        sub_boxes = [[] for _ in range(len(span_boxes))]
        for i, span_box in enumerate(span_boxes):
            for j, sub_cell in enumerate(sub_cells):
                if is_contained(span_box, sub_cell["box"]):
                    check_list[j] = False
                    sub_boxes[i].append(sub_cell)

        cells = []
        for i, sub_cell in enumerate(sub_cells):
            if check_list[i]:
                cells.append(sub_cell)

        for i, span_box in enumerate(span_boxes):
            sub_box = sub_boxes[i]

            if len(sub_box) == 0:
                continue

            row = min([box["row"] for box in sub_box])
            col = min([box["col"] for box in sub_box])
            row_span = max([box["row"] for box in sub_box]) - row + 1
            col_span = max([box["col"] for box in sub_box]) - col + 1

            span_box = list(map(int, span_box))

            cells.append(
                {
                    "col": col,
                    "row": row,
                    "col_span": col_span,
                    "row_span": row_span,
                    "box": span_box,
                }
            )

            cells = sorted(cells, key=lambda x: (x["row"], x["col"]))

        return cells, len(row_boxes), len(col_boxes)

    def __call__(self, img, table_boxes, vis=None):
        img_tensors = self.preprocess(img, table_boxes)
        outputs = []
        with torch.inference_mode():
            for data in img_tensors:
                pred = self.model(data["tensor"])
                table = self.postprocess(
                    pred,
                    data,
                )

                outputs.append(table)

        if vis is None:
            vis = img.copy()

        if self.visualize:
            for table in outputs:
                vis = table_visualizer(
                    vis,
                    table,
                )

        return outputs, vis
