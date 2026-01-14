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
from .utils.visualizer import cell_detector_visualizer

from .utils.misc import filter_by_flag, is_contained, calc_iou

from .schemas.table_semantic_parser import CellSchema, CellDetectorSchema
from .utils.misc import is_right_adjacent, is_bottom_adjacent

import numpy as np


class TableParserModelCatalog(BaseModelCatalog):
    def __init__(self):
        super().__init__()
        self.register("rtdetrv2_beta", TableParserRTDETRv2BetaConfig, RTDETRv2)


def filter_contained_rectangles_with_category(category_elements, ignore_categories=[]):
    """同一カテゴリに属する矩形のうち、他の矩形の外側に含まれるものを除外"""

    for category, elements in category_elements.items():
        if category in ignore_categories:
            continue

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

                # 双方から見て内包関係にある場合、面積の小さい方を残す
                if ij and ji:
                    if box_i_area > box_j_area:
                        check_list[i] = False
                    else:
                        check_list[j] = False

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


def find_holes_as_rects(table_shape, cell_boxes, pad=2, close_ksize=5, min_area=300):
    mask = np.full((table_shape[0], table_shape[1]), 255, np.uint8)

    for bx1, by1, bx2, by2 in cell_boxes:
        bx1, by1, bx2, by2 = map(int, [bx1, by1, bx2, by2])
        cv2.rectangle(mask, (bx1, by1), (bx2, by2), 0, thickness=-1)

    if close_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_ksize, close_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=3)

    ff = mask.copy()
    h, w = ff.shape
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, flood_mask, (0, 0), 0)

    holes = ff  # 穴だけが白として残る
    cnts, _ = cv2.findContours(holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for c in cnts:
        x, y, rw, rh = cv2.boundingRect(c)
        area = rw * rh
        if area < min_area:
            continue
        rects.append([x - pad, y - pad, x + rw + pad, y + rh + pad])

    return rects


def choose_role(role_counts):
    if not role_counts:
        return None

    max_count = max(role_counts.values())
    candidates = [r for r, c in role_counts.items() if c == max_count]

    # 同点なら必ず cell
    if len(candidates) > 1 and "cell" in candidates:
        return "cell"

    return candidates[0]


def calc_adjacent_holes_to_cells(holes, cells):
    """
    セルの隣接関係を計算
    セルが同じサブテーブル or 同じグループに属している場合のみ隣接関係を追加
    """

    directions = ["R", "L", "D", "U"]
    role = ["cell", "header", "empty"]

    kept_holes = []
    for i, hole in enumerate(holes):
        edge_counts = {dir: 0 for dir in directions}
        role_counts = {r: 0 for r in role}

        for j, node in enumerate(cells):
            # 左右の隣接判定
            if is_right_adjacent(hole["box"], node["box"]):
                edge_counts["R"] += 1
                role_counts[node["role"]] += 1
            if is_right_adjacent(node["box"], hole["box"]):
                edge_counts["L"] += 1
                role_counts[node["role"]] += 1
            # 上下の隣接判定
            if is_bottom_adjacent(hole["box"], node["box"]):
                edge_counts["D"] += 1
                role_counts[node["role"]] += 1
            if is_bottom_adjacent(node["box"], hole["box"]):
                edge_counts["U"] += 1
                role_counts[node["role"]] += 1

        if sum([count > 0 for count in edge_counts.values()]) > 2:
            hole["role"] = choose_role(role_counts)
            kept_holes.append(hole)

    return kept_holes


class CellDetector(BaseModule):
    model_catalog = TableParserModelCatalog()

    def __init__(
        self,
        model_name="rtdetrv2_beta",
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

    def preprocess(self, img, tables):
        cv_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        table_imgs = []
        for table in tables:
            x1, y1, x2, y2 = map(int, table.box)
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

        min_height = min([(cell.box[3] - cell.box[1]) for cell in cells])

        values = [c for c in cells if c.role in ["cell", "header", "empty"]]
        groups = [c for c in cells if c.role == "group"]

        values = sorted(values, key=lambda x: (x.box[1] // min_height, x.box[0]))
        groups = sorted(groups, key=lambda x: (x.box[1], x.box[0]))

        cells = values + groups
        for i, cell in enumerate(cells):
            cell.id = f"c{str(i)}"

        return cells

    def is_close_cell(self, box1, box2, threshold=10):
        x11, y11, x12, y12 = box1
        x21, y21, x22, y22 = box2

        if abs(x11 - x21) < threshold and abs(x12 - x22) < threshold:
            return True
        if abs(y11 - y21) < threshold and abs(y12 - y22) < threshold:
            return True
        return False

    def is_fully_contained(self, box1, box2, threshold=0.9):
        overlap_ratio = calc_iou(box1, box2)
        return overlap_ratio >= threshold

    def postprocess(self, preds, data, table_box):
        h, w = data["size"]
        orig_size = torch.tensor([w, h])[None].to(self.device)
        outputs = self.postprocessor(preds, orig_size, self.thresh_score)

        preds = outputs[0]

        boxes = preds["boxes"]
        labels = preds["labels"]
        scores = preds["scores"]

        category_elements = {category: [] for category in self.label_mapper.values()}
        category_elements["hole"] = []

        for box, score, label in zip(boxes, scores, labels):
            category = self.label_mapper[label.item()]
            box = box.astype(int).tolist()
            if self.is_fully_contained(box, [0, 0, w, h]):
                continue

            category_elements[category].append(
                {
                    "box": box,
                    "score": float(score),
                    "role": category,
                }
            )

        category_elements = filter_contained_rectangles_with_category(
            category_elements,
            ignore_categories=["group"],
        )
        category_elements = filter_contained_rectangles_across_categories(
            category_elements,
            source="header",
            target="cell",
        )

        cell_boxes = (
            category_elements["cell"]
            + category_elements["header"]
            + category_elements["empty"]
        )

        hole_boxes = find_holes_as_rects(
            data["size"],
            [cell["box"] for cell in cell_boxes],
        )

        for box in hole_boxes:
            category_elements["hole"].append(
                {
                    "box": box,
                    "score": 1.0,
                    "role": "hole",
                }
            )

        for category, cells in category_elements.items():
            for cell in cells:
                cell["box"][0] += data["offset"][0]
                cell["box"][1] += data["offset"][1]
                cell["box"][2] += data["offset"][0]
                cell["box"][3] += data["offset"][1]

        # グループが検出されなかった場合、テーブル全体をグループとして扱う
        if len(category_elements["group"]) == 0:
            category_elements["group"] = [
                {
                    "box": table_box,
                    "role": "group",
                }
            ]

        # テーブル内にセルが検出されなかった場合、テーブル全体をセルとして扱う
        if (
            len(
                category_elements["cell"]
                + category_elements["empty"]
                + category_elements["header"]
            )
            == 0
        ):
            category_elements["cell"] = [
                {
                    "box": table_box,
                    "role": "cell",
                }
            ]

        table_x, table_y = data["offset"]
        table_x2 = table_x + data["size"][1]
        table_y2 = table_y + data["size"][0]
        table_box = [table_x, table_y, table_x2, table_y2]

        cells = self.extract_cell_elements(category_elements)
        cells = self.remove_noise_cells(cells, min_width=10, min_height=10)
        cells = self.sort_cells(cells)
        return cells

    def remove_noise_cells(self, cells, min_width=30, min_height=30):
        filtered_cells = []
        for cell in cells:
            box = cell.box
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            if width > min_width and height > min_height:
                filtered_cells.append(cell)
        return filtered_cells

    def extract_cell_elements(self, elements):
        elements["hole"] = calc_adjacent_holes_to_cells(
            elements["hole"],
            [c for c in elements["cell"] + elements["header"] + elements["empty"]],
        )

        cells = []
        for category, values in elements.items():
            if category in ["cell", "header", "empty", "group", "hole"]:
                for value in values:
                    box = value["box"]
                    cells.append(
                        CellSchema(
                            id=None,
                            box=box,
                            role=value["role"],
                            contents=None,
                            row=None,
                            col=None,
                            row_span=None,
                            col_span=None,
                        )
                    )

        return cells

    def __call__(self, img, tables, vis=None):
        img_tensors = self.preprocess(img, tables)
        outputs = []
        for _, (data, table) in enumerate(zip(img_tensors, tables)):
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

            cells = self.postprocess(pred, data, table.box)

            if len(cells) == 0:
                continue

            outputs.append(
                CellDetectorSchema(
                    id=f"t{str(table.order)}",
                    box=table.box,
                    role=table.role,
                    cells=cells,
                )
            )

        vis_cell = vis.copy()
        vis_group = vis.copy()

        if self.visualize:
            for table in outputs:
                vis_cell, vis_group = cell_detector_visualizer(
                    vis_cell,
                    vis_group,
                    table.cells,
                )

        return outputs, vis_cell, vis_group
