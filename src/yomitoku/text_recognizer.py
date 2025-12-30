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
from .data.dataset import ParseqDataset
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
        infer_torch_script=False,
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

        if infer_torch_script:
            name = self._cfg.hf_hub_repo.split("/")[-1]
            path_ts = f"{ROOT_DIR}/torch_script/{name}-{self.device}.pt"
            if not os.path.exists(path_ts):
                self.convert_torch_script(path_ts)

            self.model = torch.jit.load(path_ts, map_location=self.device)
            # self.model = torch.jit.optimize_for_inference(self.model)
            self.model.eval()

        if self.model is not None:
            print(self.device)
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

        return dataloader, polygons

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

    def convert_torch_script(self, path_ts):
        img_size = self._cfg.data.img_size
        batch_size = self._cfg.data.batch_size
        dummy_input = torch.randn(batch_size, 3, *img_size, requires_grad=False).to(
            self.device
        )
        ts = torch.jit.trace(
            self.model.eval().to(self.device),
            dummy_input,
            strict=False,
        )
        # ts = torch.jit.freeze(ts)
        ts.save(path_ts)
        # traced_script_module.save(path_ts)

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

    def __call__(self, img, points=None, vis=None):
        """
        Apply the recognition model to the input image.

        Args:
            img (np.ndarray): target image(BGR)
            points (list): list of quadrilaterals. Each quadrilateral is represented as a list of 4 points sorted clockwise.
            vis (np.ndarray, optional): rendering image. Defaults to None.
        """

        dataloader, points = self.preprocess(img, points)
        preds = []
        scores = []
        directions = []
        for data in dataloader:
            if self.infer_onnx:
                input = data.numpy()
                results = self.sess.run(["output"], {"input": input})
                p = torch.tensor(results[0])
            else:
                with torch.inference_mode():
                    data = data.to(self.device)
                    print(data.shape)
                    p = self.model(data).softmax(-1)

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
