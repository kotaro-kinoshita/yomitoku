import cv2
import os
import onnx
import onnxruntime
import torch
import torchvision.transforms as T
from PIL import Image


from .constants import ROOT_DIR

from .base import BaseModelCatalog, BaseModule
from .configs import TableParserRTDETRv2BetaConfig
from .models import RTDETRv2
from .postprocessor import RTDETRPostProcessor
from .utils.visualizer import table_parser_visualizer

from .utils.misc import calc_overlap_ratio


class TableParserModelCatalog(BaseModelCatalog):
    def __init__(self):
        super().__init__()
        self.register("rtdetrv2beta", TableParserRTDETRv2BetaConfig, RTDETRv2)


class TableParser(BaseModule):
    model_catalog = TableParserModelCatalog()

    def __init__(
        self,
        model_name="rtdetrv2beta",
        path_cfg=None,
        device="cuda",
        visualize=False,
        from_pretrained=True,
        infer_onnx=False,
    ):
        super().__init__()
        self.load_model(
            model_name,
            path_cfg,
            from_pretrained=from_pretrained,
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
                    model.SerializeToString(),
                    providers=["CUDAExecutionProvider"],
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

    def preprocess(self, img, boxes):
        cv_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        table_imgs = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            table_img = cv_img[y1:y2, x1:x2, :]
            th, hw = table_img.shape[:2]
            table_img = Image.fromarray(table_img)
            img_tensor = self.transforms(table_img)[None]
            table_imgs.append(
                {
                    "tensor": img_tensor,
                    "size": (th, hw),
                    "offset": (x1, y1),
                }
            )
        return table_imgs

    def sort_cells(self, cells):
        if len(cells) == 0:
            return cells

        min_height = min([(cell["box"][3] - cell["box"][1]) for cell in cells])

        values = [c for c in cells if c["role"] in ["cell", "header", "empty"]]
        groups = [c for c in cells if c["role"] == "group"]

        values = sorted(values, key=lambda x: (x["box"][1] // min_height, x["box"][0]))
        groups = sorted(groups, key=lambda x: (x["box"][1], x["box"][0]))

        cells = values + groups
        for i, cell in enumerate(cells):
            cell["id"] = i + 1

        return cells

    def is_close_cell(self, box1, box2, threshold=10):
        x11, y11, x12, y12 = box1
        x21, y21, x22, y22 = box2

        if abs(x11 - x21) < threshold and abs(x12 - x22) < threshold:
            return True
        if abs(y11 - y21) < threshold and abs(y12 - y22) < threshold:
            return True
        return False

    def postprocess(self, preds, data, table_box):
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

            if (
                calc_overlap_ratio(
                    box,
                    table_box,
                )[0]
                > 0.95
            ):
                continue

            category_elements[category].append(
                {
                    "box": box,
                    "score": float(score),
                }
            )

        # category_elements = filter_contained_rectangles_within_category(
        #    category_elements, ["group"]
        # )

        # category_elements = filter_contained_rectangles_across_categories(
        #    category_elements, "cell", "header"
        # )

        table_x, table_y = data["offset"]
        table_x2 = table_x + data["size"][1]
        table_y2 = table_y + data["size"][0]
        table_box = [table_x, table_y, table_x2, table_y2]

        cells = self.extract_cell_elements(category_elements)
        cells = self.remove_noize_cells(cells, min_width=10, min_height=10)
        print(cells)
        cells = self.sort_cells(cells)

        if len(cells) == 0:
            cells = [
                {
                    "col": [],
                    "row": [],
                    "col_span": 1,
                    "row_span": 1,
                    "box": table_box,
                    "role": "cell",
                    "contents": None,
                    "id": 1,
                }
            ]
        table = {
            "box": table_box,
            "n_row": 0,
            "n_col": 0,
            "rows": [],
            "cols": [],
            "spans": [],
            "cells": cells,
            "order": 0,
        }

        # results = TableStructureRecognizerSchema(**table)
        return table

    def remove_noize_cells(self, cells, min_width=30, min_height=30):
        filtered_cells = []
        for cell in cells:
            box = cell["box"]
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            if width > min_width and height > min_height:
                filtered_cells.append(cell)
        return filtered_cells

    def extract_cell_elements(self, elements):
        cells = []
        for category, values in elements.items():
            if category in ["cell", "header", "empty", "group"]:
                # if category in ["group"]:
                for value in values:
                    box = value["box"]
                    cells.append(
                        {
                            "col": [],
                            "row": [],
                            "col_span": 1,
                            "row_span": 1,
                            "box": box,
                            "role": category,
                            "contents": None,
                        }
                    )

        return cells

    def __call__(self, img, table_boxes, vis=None):
        img_tensors = self.preprocess(img, table_boxes)
        outputs = []
        for data, box in zip(img_tensors, table_boxes):
            if self.infer_onnx:
                input = data["tensor"].numpy()
                results = self.sess.run(None, {"input": input})
                pred = {
                    "pred_logits": torch.tensor(results[0]).to(self.device),
                    "pred_boxes": torch.tensor(results[1]).to(self.device),
                }

            else:
                with torch.inference_mode():
                    data["tensor"] = data["tensor"].to(self.device)
                    pred = self.model(data["tensor"])

            table = self.postprocess(pred, data, box)

            if len(table["cells"]) > 0:
                outputs.append(table)

        vis_cell = vis.copy()
        vis_group = vis.copy()

        if self.visualize:
            for table in outputs:
                vis_cell, vis_group = table_parser_visualizer(
                    vis_cell,
                    vis_group,
                    table,
                )

        return outputs, vis_cell, vis_group
