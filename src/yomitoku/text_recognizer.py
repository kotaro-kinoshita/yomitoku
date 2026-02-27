import numpy as np
import torch
import os
import unicodedata

from .base import BaseModelCatalog, BaseModule
from .configs import (
    TextRecognizerPARSeqConfig,
    TextRecognizerPARSeqSmallConfig,
    TextRecognizerPARSeqV2Config,
    TextRecognizerPARSeqTinyConfig,
)
import cv2

from .data.dataset import ParseqDataset
from .data.functions import resize_with_padding
from .models import PARSeq
from .postprocessor import ParseqTokenizer as Tokenizer
from .utils.misc import load_charset
from .utils.visualizer import rec_visualizer

from .constants import ROOT_DIR
from .schemas import TextRecognizerSchema

import onnx
import onnxruntime


class TextRecognizerModelCatalog(BaseModelCatalog):
    def __init__(self):
        super().__init__()
        self.register("parseq", TextRecognizerPARSeqConfig, PARSeq)
        self.register("parseqv2", TextRecognizerPARSeqV2Config, PARSeq)
        self.register("parseq-small", TextRecognizerPARSeqSmallConfig, PARSeq)
        self.register("parseq-tiny", TextRecognizerPARSeqTinyConfig, PARSeq)


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
        rec_orientation_fallback=True,
        rec_orientation_fallback_thresh=0.85,
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
        self.rec_orientation_fallback = rec_orientation_fallback
        self.rec_orientation_fallback_thresh = rec_orientation_fallback_thresh

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

    def preprocess(self, img, polygons):
        if polygons is None:
            h, w = img.shape[:2]
            polygons = [
                [
                    [0, 0],
                    [w, 0],
                    [w, h],
                    [0, h],
                ]
            ]

        dataset = ParseqDataset(self._cfg, img, polygons)
        dataloader = self._make_mini_batch(dataset)

        return dataloader, polygons, dataset

    def _make_mini_batch(self, dataset):
        mini_batches = []
        mini_batch = []
        for data in dataset:
            data = torch.unsqueeze(data, 0)
            mini_batch.append(data)

            if len(mini_batch) == self._cfg.data.batch_size:
                mini_batches.append(torch.cat(mini_batch, 0))
                mini_batch = []
        else:
            if len(mini_batch) > 0:
                mini_batches.append(torch.cat(mini_batch, 0))

        return mini_batches

    def _make_mini_batch_from_tensor(self, tensor):
        mini_batches = []
        batch_size = self._cfg.data.batch_size
        for i in range(0, len(tensor), batch_size):
            mini_batches.append(tensor[i : i + batch_size])
        return mini_batches

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
            opset_version=16,
            input_names=["input"],
            output_names=["output"],
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
        )

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

    def _run_inference(self, data):
        if self.infer_onnx:
            input = data.numpy()
            results = self.sess.run(["output"], {"input": input})
            p = torch.tensor(results[0])
        else:
            with torch.inference_mode():
                data = data.to(self.device)
                p = self.model(data).softmax(-1)
        return p

    def _run_batch_inference(self, dataloader, points):
        preds = []
        scores = []
        directions = []
        offset = 0
        for data in dataloader:
            batch_points = points[offset : offset + len(data)]
            p = self._run_inference(data)
            pred, score, direction = self.postprocess(p, batch_points)
            preds.extend(pred)
            scores.extend(score)
            directions.extend(direction)
            offset += len(data)
        return preds, scores, directions

    def _prepare_fallback_batch(self, dataset, indices):
        img_size = self._cfg.data.img_size
        tensors = []
        for i in indices:
            rotated = cv2.rotate(dataset.roi_images[i], cv2.ROTATE_180)
            resized = resize_with_padding(rotated, img_size)
            tensor = dataset.transform(resized).unsqueeze(0)
            tensors.append(tensor)
        batch = torch.cat(tensors, dim=0)
        return self._make_mini_batch_from_tensor(batch)

    def _apply_orientation_fallback(self, dataset, points, preds, scores, directions):
        retry_indices = [
            i for i, s in enumerate(scores) if s < self.rec_orientation_fallback_thresh
        ]
        if len(retry_indices) == 0:
            return

        retry_points = [points[i] for i in retry_indices]
        retry_dataloader = self._prepare_fallback_batch(dataset, retry_indices)
        retry_preds, retry_scores, retry_directions = self._run_batch_inference(
            retry_dataloader, retry_points
        )

        for j, idx in enumerate(retry_indices):
            if retry_scores[j] > scores[idx]:
                preds[idx] = retry_preds[j]
                scores[idx] = retry_scores[j]
                directions[idx] = retry_directions[j]

    def __call__(self, img, points=None, vis=None):
        """
        Apply the recognition model to the input image.

        Args:
            img (np.ndarray): target image(BGR)
            points (list): list of quadrilaterals. Each quadrilateral is represented as a list of 4 points sorted clockwise.
            vis (np.ndarray, optional): rendering image. Defaults to None.
        """

        dataloader, points, dataset = self.preprocess(img, points)
        preds, scores, directions = self._run_batch_inference(dataloader, points)

        if self.rec_orientation_fallback:
            self._apply_orientation_fallback(dataset, points, preds, scores, directions)

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
