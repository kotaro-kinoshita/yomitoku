import numpy as np
import torch
import os

from .base import BaseModelCatalog, BaseModule
from .configs import (
    TextDetectorDBNetConfig,
    TextDetectorDBNetV2Config,
)
from .data.functions import (
    array_to_tensor,
    resize_shortest_edge,
    standardization_image,
)
from .models import DBNet
from .postprocessor import DBnetPostProcessor
from .utils.visualizer import det_visualizer
from .constants import ROOT_DIR
from .schemas import TextDetectorSchema

import onnx
import onnxruntime


class TextDetectorModelCatalog(BaseModelCatalog):
    def __init__(self):
        super().__init__()
        self.register("dbnet", TextDetectorDBNetConfig, DBNet)
        self.register("dbnetv2", TextDetectorDBNetV2Config, DBNet)


class TextDetector(BaseModule):
    model_catalog = TextDetectorModelCatalog()

    def __init__(
        self,
        model_name="dbnetv2",
        path_cfg=None,
        device="cuda",
        visualize=False,
        from_pretrained=True,
        infer_onnx=False,
        infer_torch_script=True,
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
        self.post_processor = DBnetPostProcessor(**self._cfg.post_process)
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

            self.model = None

        if infer_torch_script:
            name = self._cfg.hf_hub_repo.split("/")[-1]
            path_ts = f"{ROOT_DIR}/torch_script/{name}-{self.device}.pt"
            if not os.path.exists(path_ts):
                self.convert_torch_script(path_ts)

            self.model = torch.jit.load(path_ts, map_location=self.device)
            self.model = torch.jit.optimize_for_inference(self.model)
            self.model.eval()

        if self.model is not None:
            print(self.device)
            self.model.to(self.device)

    def convert_onnx(self, path_onnx):
        dynamic_axes = {
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"},
        }

        dummy_input = torch.randn(1, 3, 256, 256, requires_grad=True)

        torch.onnx.export(
            self.model,
            dummy_input,
            path_onnx,
            opset_version=16,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )

    def convert_torch_script(self, path_ts):
        dummy_input = torch.randn(1, 3, 256, 256, requires_grad=False).to(self.device)
        ts = torch.jit.trace(
            self.model.to(self.device),
            dummy_input,
            strict=False,
        )
        ts = torch.jit.freeze(ts)
        ts.save(path_ts)
        # traced_script_module.save(path_ts)

    def preprocess(self, img):
        img = img.copy()
        img = img[:, :, ::-1].astype(np.float32)
        resized = resize_shortest_edge(
            img, self._cfg.data.shortest_size, self._cfg.data.limit_size
        )
        normalized = standardization_image(resized)
        tensor = array_to_tensor(normalized)
        return tensor

    def postprocess(self, preds, image_size):
        return self.post_processor(preds, image_size)

    def __call__(self, img):
        """apply the detection model to the input image.

        Args:
            img (np.ndarray): target image(BGR)
        """

        ori_h, ori_w = img.shape[:2]
        tensor = self.preprocess(img)

        if self.infer_onnx:
            input = tensor.numpy()
            results = self.sess.run(["output"], {"input": input})
            preds = {"binary": torch.tensor(results[0])}
        else:
            with torch.inference_mode():
                print("aaa")
                tensor = tensor.to(self.device)
                preds = self.model(tensor)

        quads, scores = self.postprocess(preds, (ori_h, ori_w))
        outputs = {"points": quads, "scores": scores}

        results = TextDetectorSchema(**outputs)

        vis = None
        if self.visualize:
            vis = det_visualizer(
                img,
                quads,
                preds=preds,
                vis_heatmap=self._cfg.visualize.heatmap,
                line_color=tuple(self._cfg.visualize.color[::-1]),
            )

        return results, vis
