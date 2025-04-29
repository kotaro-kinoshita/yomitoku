import cv2
import math
import numpy as np
import pyclipper
from shapely.geometry import Polygon


class DBnetPostProcessor:
    def __init__(self, min_size, thresh, box_thresh, max_candidates, unclip_ratio):
        self.min_size = min_size
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio

    def __call__(self, preds, batch_ori_h, batch_ori_w):
        """
        pred:
            binary: text region segmentation map, with shape (N, H, W)
            thresh: [if exists] thresh hold prediction with shape (N, H, W)
            thresh_binary: [if exists] binarized with threshhold, (N, H, W)
        """

        batch_quads, batch_scores = [], []
        for pred, height, width in zip(preds["binary"], batch_ori_h, batch_ori_w):
            segmentation = self.binarize(pred)[0]
            quads, scores = self.boxes_from_bitmap(pred, segmentation, width, height)
            batch_quads.append(quads)
            batch_scores.append(scores)

        return (
            batch_quads,
            batch_scores,
        )

    def binarize(self, pred):
        return pred > self.thresh

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        """
        _bitmap: single map with shape (H, W),
            whose values are binarized as {0, 1}
        """

        assert len(_bitmap.shape) == 2
        bitmap = _bitmap.cpu().numpy()  # The first channel

        pred = pred.cpu().detach().numpy()[0]
        height, width = bitmap.shape
        contours, _ = cv2.findContours(
            (bitmap * 255).astype(np.uint8),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index].squeeze(1)
            points, sside = self.get_mini_boxes(contour)

            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, contour)

            if self.box_thresh > score:
                continue

            box = self.unclip(points, unclip_ratio=self.unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height
            )

            boxes.append(box.astype(np.int16).tolist())
            scores.append(score)

        return boxes, scores

    def unclip(self, box, unclip_ratio=7):
        # 小さい文字が見切れやすい、大きい文字のマージンが過度に大きくなる等の課題がある
        # 対応として、文字の大きさに応じて、拡大パラメータを動的に変更する
        # Note: こののルールはヒューリスティックで理論的根拠はない
        poly = Polygon(box)
        width = box[:, 0].max() - box[:, 0].min()
        height = box[:, 1].max() - box[:, 1].min()
        box_dist = min(width, height)
        ratio = unclip_ratio / math.sqrt(box_dist)

        distance = poly.area * ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [
            points[index_1],
            points[index_2],
            points[index_3],
            points[index_4],
        ]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]
