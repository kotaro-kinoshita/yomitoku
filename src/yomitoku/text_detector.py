import numpy as np
import torch
import os

from .base import BaseModelCatalog, BaseModule
from .configs import (
    TextDetectorDBNetConfig,
    TextDetectorDBNetV2Config,
)
from .data.functions import (
    resize_shortest_edge,
)
from .models import DBNet
from .postprocessor import DBnetPostProcessor
from .utils.visualizer import det_visualizer
from .constants import ROOT_DIR
from .schemas import TextDetectorSchema

import onnx
import onnxruntime


class TextDetectorTransform:
    """GPU-accelerated preprocessing for TextDetector."""

    def __init__(self, device, shortest_size, limit_size):
        self.device = device
        self.shortest_size = shortest_size
        self.limit_size = limit_size
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self._device_ready = False

    def _ensure_device(self):
        if not self._device_ready:
            self.mean = self.mean.to(self.device)
            self.std = self.std.to(self.device)
            self._device_ready = True

    def __call__(self, img: np.ndarray) -> torch.Tensor:
        self._ensure_device()

        # Note: Original code had double BGR/RGB flip that canceled out.
        # Keeping BGR format for bug-compatibility with trained model.
        # 1. Resize on CPU (cv2 is efficient)
        resized = resize_shortest_edge(img, self.shortest_size, self.limit_size)

        # 2. Transfer to GPU + normalize (GPU-accelerated)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(device=self.device, dtype=torch.float32)
        tensor = (tensor / 255.0 - self.mean) / self.std

        return tensor


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

        if self.model is not None:
            self.model.to(self.device)

        self._transform = None  # Lazy initialization

    def _get_transform(self):
        if self._transform is None:
            self._transform = TextDetectorTransform(
                self.device,
                self._cfg.data.shortest_size,
                self._cfg.data.limit_size,
            )
        return self._transform

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

    def preprocess(self, img):
        return self._get_transform()(img)

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
            input_np = tensor.cpu().numpy()  # Move from GPU to CPU for ONNX
            results = self.sess.run(["output"], {"input": input_np})
            preds = {"binary": torch.tensor(results[0])}
        else:
            with torch.inference_mode():
                preds = self.model(tensor)  # tensor is already on device

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
