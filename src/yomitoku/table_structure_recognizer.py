from typing import List, Union

import cv2
import torch
import torchvision.transforms as T
from PIL import Image
from pydantic import conlist

from .base import BaseModelCatalog, BaseModule, BaseSchema
from .configs import TableStructureRecognizerRTDETRv2Config
from .layout_parser import filter_contained_rectangles_within_category
from .models import RTDETRv2
from .postprocessor import RTDETRPostProcessor
from .utils.misc import calc_intersection, filter_by_flag, is_contained
from .utils.visualizer import table_visualizer


class TableStructureRecognizerModelCatalog(BaseModelCatalog):
    def __init__(self):
        super().__init__()
        self.register("rtdetrv2", TableStructureRecognizerRTDETRv2Config, RTDETRv2)


class TableCellSchema(BaseSchema):
    col: int
    row: int
    col_span: int
    row_span: int
    box: conlist(int, min_length=4, max_length=4)
    contents: Union[str, None]


class TableStructureRecognizerSchema(BaseSchema):
    box: conlist(int, min_length=4, max_length=4)
    n_row: int
    n_col: int
    cells: List[TableCellSchema]
    order: int


def extract_cells(row_boxes, col_boxes):
    cells = []
    for i, row_box in enumerate(row_boxes):
        for j, col_box in enumerate(col_boxes):
            intersection = calc_intersection(row_box, col_box)
            if intersection is None:
                continue

            cells.append(
                {
                    "col": j + 1,
                    "row": i + 1,
                    "col_span": 1,
                    "row_span": 1,
                    "box": intersection,
                    "contents": None,
                }
            )

    return cells


def filter_contained_cells_within_spancell(cells, span_boxes):
    check_list = [True] * len(cells)
    child_boxes = [[] for _ in range(len(span_boxes))]
    for i, span_box in enumerate(span_boxes):
        for j, sub_cell in enumerate(cells):
            if is_contained(span_box, sub_cell["box"]):
                check_list[j] = False
                child_boxes[i].append(sub_cell)

    cells = filter_by_flag(cells, check_list)

    for i, span_box in enumerate(span_boxes):
        child_box = child_boxes[i]

        if len(child_box) == 0:
            continue

        row = min([box["row"] for box in child_box])
        col = min([box["col"] for box in child_box])
        row_span = max([box["row"] for box in child_box]) - row + 1
        col_span = max([box["col"] for box in child_box]) - col + 1

        span_box = list(map(int, span_box))

        cells.append(
            {
                "col": col,
                "row": row,
                "col_span": col_span,
                "row_span": row_span,
                "box": span_box,
                "contents": None,
            }
        )

    cells = sorted(cells, key=lambda x: (x["row"], x["col"]))
    return cells


class TableStructureRecognizer(BaseModule):
    model_catalog = TableStructureRecognizerModelCatalog()

    def __init__(
        self,
        model_name="rtdetrv2",
        path_cfg=None,
        device="cuda",
        visualize=False,
        from_pretrained=True,
    ):
        super().__init__()
        self.load_model(
            model_name,
            path_cfg,
            from_pretrained=True,
        )
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

        self.label_mapper = {
            id: category for id, category in enumerate(self._cfg.category)
        }

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
        outputs = self.postprocessor(preds, orig_size, self.thresh_score)

        preds = outputs[0]
        scores = preds["scores"]
        boxes = preds["boxes"]
        labels = preds["labels"]

        category_elements = {category: [] for category in self.label_mapper.values()}
        for box, score, label in zip(boxes, scores, labels):
            category = self.label_mapper[label.item()]
            box = box.astype(int).tolist()

            box[0] += data["offset"][0]
            box[1] += data["offset"][1]
            box[2] += data["offset"][0]
            box[3] += data["offset"][1]

            category_elements[category].append(
                {
                    "box": box,
                    "score": float(score),
                }
            )

        category_elements = filter_contained_rectangles_within_category(
            category_elements
        )

        cells, n_row, n_col = self.extract_cell_elements(category_elements)

        table_x, table_y = data["offset"]
        table_x2 = table_x + data["size"][1]
        table_y2 = table_y + data["size"][0]
        table_box = [table_x, table_y, table_x2, table_y2]

        table = {
            "box": table_box,
            "n_row": n_row,
            "n_col": n_col,
            "cells": cells,
            "order": 0,
        }

        results = TableStructureRecognizerSchema(**table)

        return results

    def extract_cell_elements(self, elements):
        row_boxes = [element["box"] for element in elements["row"]]
        col_boxes = [element["box"] for element in elements["col"]]
        span_boxes = [element["box"] for element in elements["span"]]

        row_boxes = sorted(row_boxes, key=lambda x: x[1])
        col_boxes = sorted(col_boxes, key=lambda x: x[0])

        cells = extract_cells(row_boxes, col_boxes)
        cells = filter_contained_cells_within_spancell(cells, span_boxes)

        return cells, len(row_boxes), len(col_boxes)

    def __call__(self, img, table_boxes, vis=None):
        img_tensors = self.preprocess(img, table_boxes)
        outputs = []
        for data in img_tensors:
            with torch.inference_mode():
                pred = self.model(data["tensor"])
            table = self.postprocess(pred, data)
            outputs.append(table)

        if vis is None and self.visualize:
            vis = img.copy()

        if self.visualize:
            for table in outputs:
                vis = table_visualizer(
                    vis,
                    table,
                )

        return outputs, vis
