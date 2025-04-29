from typing import List

import torch
import os
import unicodedata
from pydantic import conlist

from .base import BaseModelCatalog, BaseModule, BaseSchema
from .configs import (
    TextRecognizerPARSeqConfig,
    TextRecognizerPARSeqSmallConfig,
    TextRecognizerPARSeqV2Config,
)
from .data.dataset import ParseqDataset
from .models import PARSeq
from .postprocessor import ParseqTokenizer as Tokenizer
from .utils.misc import load_charset
from .utils.visualizer import rec_visualizer

from .constants import ROOT_DIR
import onnx
import onnxruntime


class TextRecognizerModelCatalog(BaseModelCatalog):
    def __init__(self):
        super().__init__()
        self.register("parseq", TextRecognizerPARSeqConfig, PARSeq)
        self.register("parseqv2", TextRecognizerPARSeqV2Config, PARSeq)
        self.register("parseq-small", TextRecognizerPARSeqSmallConfig, PARSeq)


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
        model_name="parseqv2",
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
        self.charset = load_charset(self._cfg.charset)
        self.tokenizer = Tokenizer(self.charset)

        self.device = device

        self.model.tokenizer = self.tokenizer
        self.model.eval()

        self.visualize = visualize

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

    def preprocess(self, imgs, polygons):
        dataset = ParseqDataset(self._cfg, imgs, polygons)
        dataloader, indies, directions = self._make_mini_batch(dataset)

        return dataloader, indies, directions

    def _make_mini_batch(self, dataset):
        mini_batches = []
        mini_batch = []
        indies = []
        directions = []
        for data in dataset:
            tensor = torch.unsqueeze(data["tensor"], 0)
            mini_batch.append(tensor)
            indies.append(data["img_idx"])
            directions.append(data["direction"])

            if len(mini_batch) == self._cfg.data.batch_size:
                mini_batches.append(torch.cat(mini_batch, 0))
                mini_batch = []
        else:
            if len(mini_batch) > 0:
                mini_batches.append(torch.cat(mini_batch, 0))

        return mini_batches, indies, directions

    def convert_onnx(self, path_onnx):
        img_size = self._cfg.data.img_size
        input = torch.randn(1, 3, *img_size, requires_grad=True)
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

        self.model.export_onnx = True
        torch.onnx.export(
            self.model,
            input,
            path_onnx,
            opset_version=14,
            input_names=["input"],
            output_names=["output"],
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
        )

    def postprocess(self, p, points):
        pred, score = self.tokenizer.decode(p)
        pred = [unicodedata.normalize("NFKC", x) for x in pred]

        return pred, score

    def __call__(self, imgs, batch_points, vis_imgs=None):
        """
        Apply the recognition model to the input image.

        Args:
            img (np.ndarray): target image(BGR)
            points (list): list of quadrilaterals. Each quadrilateral is represented as a list of 4 points sorted clockwise.
            vis (np.ndarray, optional): rendering image. Defaults to None.
        """

        dataloader, indies, directions = self.preprocess(imgs, batch_points)
        preds = []
        scores = []
        for data in dataloader:
            if self.infer_onnx:
                input = data.numpy()
                results = self.sess.run(["output"], {"input": input})
                p = torch.tensor(results[0])
            else:
                with torch.inference_mode():
                    data = data.to(self.device)
                    p = self.model(data).softmax(-1)

            pred, score = self.postprocess(p, batch_points)
            preds.extend(pred)
            scores.extend(score)

        points = [quad for points in batch_points for quad in points]

        outputs = {}
        for pred, score, point, direction, index in zip(
            preds,
            scores,
            points,
            directions,
            indies,
        ):
            if index not in outputs:
                outputs[index] = {
                    "contents": [],
                    "scores": [],
                    "points": [],
                    "directions": [],
                }

            outputs[index]["contents"].append(pred)
            outputs[index]["scores"].append(score)
            outputs[index]["points"].append(point)
            outputs[index]["directions"].append(direction)

        results = []
        visualize_imgs = []
        for output, img, vis in zip(outputs.values(), imgs, vis_imgs):
            result = TextRecognizerSchema(**output)
            results.append(result)

            if self.visualize:
                if vis is None:
                    vis = img.copy()
                vis = rec_visualizer(
                    vis,
                    result,
                    font_size=self._cfg.visualize.font_size,
                    font_color=tuple(self._cfg.visualize.color[::-1]),
                    font_path=self._cfg.visualize.font,
                )

                visualize_imgs.append(vis)
        return results, visualize_imgs
