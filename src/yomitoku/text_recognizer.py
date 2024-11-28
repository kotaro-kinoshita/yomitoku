from typing import List

import numpy as np
import torch
import unicodedata
from pydantic import conlist

from .base import BaseModelCatalog, BaseModule, BaseSchema
from .configs import TextRecognizerPARSeqConfig
from .data.dataset import ParseqDataset
from .models import PARSeq
from .postprocessor import ParseqTokenizer as Tokenizer
from .utils.misc import load_charset
from .utils.visualizer import rec_visualizer


class TextRecognizerModelCatalog(BaseModelCatalog):
    def __init__(self):
        super().__init__()
        self.register("parseq", TextRecognizerPARSeqConfig, PARSeq)


class TextRecognizerSchema(BaseSchema):
    contents: List[str]
    directions: List[str]
    scores: List[float]
    points: List[
        conlist(
            conlist(int, min_length=2, max_length=2),
            min_length=4,
            max_length=4,
        )
    ]


class TextRecognizer(BaseModule):
    model_catalog = TextRecognizerModelCatalog()

    def __init__(
        self,
        model_name="parseq",
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
        self.charset = load_charset(self._cfg.charset)
        self.tokenizer = Tokenizer(self.charset)

        self.device = device

        self.model.eval()
        self.model.to(self.device)

        self.visualize = visualize

    def preprocess(self, img, polygons):
        dataset = ParseqDataset(self._cfg, img, polygons)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self._cfg.data.batch_size,
            shuffle=False,
            num_workers=self._cfg.data.num_workers,
        )

        return dataloader

    def postprocess(self, p, points):
        pred, score = self.tokenizer.decode(p)
        pred = [unicodedata.normalize("NFKC", x) for x in pred]

        directions = []
        for point in points:
            point = np.array(point)
            w = np.linalg.norm(point[0] - point[1])
            h = np.linalg.norm(point[1] - point[2])

            direction = "vertical" if h > w * 2 else "horizontal"
            directions.append(direction)

        return pred, score, directions

    def __call__(self, img, points, vis=None):
        """
        Apply the recognition model to the input image.

        Args:
            img (np.ndarray): target image(BGR)
            points (list): list of quadrilaterals. Each quadrilateral is represented as a list of 4 points sorted clockwise.
            vis (np.ndarray, optional): rendering image. Defaults to None.
        """

        dataloader = self.preprocess(img, points)
        preds = []
        scores = []
        directions = []
        for data in dataloader:
            data = data.to(self.device)
            with torch.inference_mode():
                p = self.model(self.tokenizer, data).softmax(-1)
                pred, score, direction = self.postprocess(p, points)
                preds.extend(pred)
                scores.extend(score)
                directions.extend(direction)

        outputs = {
            "contents": preds,
            "scores": scores,
            "points": points,
            "directions": directions,
        }
        results = TextRecognizerSchema(**outputs)

        if self.visualize:
            if vis is None:
                vis = img.copy()
            vis = rec_visualizer(
                vis,
                results,
                font_size=self._cfg.visualize.font_size,
                font_color=tuple(self._cfg.visualize.color[::-1]),
                font_path=self._cfg.visualize.font,
            )

        return results, vis
