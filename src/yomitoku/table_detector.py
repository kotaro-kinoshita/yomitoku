import cv2
import os
import onnx
import onnxruntime
import torch
import torchvision.transforms as T
from PIL import Image

from .constants import ROOT_DIR

from .base import BaseModelCatalog, BaseModule
from .configs import TableDetectorRTDETRv2BetaConfig

from .models import RTDETRv2
from .postprocessor import RTDETRPostProcessor
from .utils.visualizer import table_detector_visualizer
from .reading_order import prediction_reading_order
from .layout_parser import filter_contained_rectangles_within_category

from .schemas.document_analyzer import Element


class TableDetectorModelCatalog(BaseModelCatalog):
    def __init__(self):
        super().__init__()
        self.register("rtdetrv2_beta", TableDetectorRTDETRv2BetaConfig, RTDETRv2)


class TableDetector(BaseModule):
    model_catalog = TableDetectorModelCatalog()

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
        self.load_model(model_name, path_cfg, from_pretrained)
        self.device = device
        self.visualize = visualize

        self.model.eval()
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

        self.role = self._cfg.role
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

    def preprocess(self, img):
        cv_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv_img)
        img_tensor = self.transforms(img)[None]
        return img_tensor

    def postprocess(self, preds, image_size):
        h, w = image_size
        orig_size = torch.tensor([w, h])[None].to(self.device)
        outputs = self.postprocessor(preds, orig_size, self.thresh_score)
        outputs = self.filtering_elements(outputs[0])

        return outputs

    def filtering_elements(self, preds):
        scores = preds["scores"]
        boxes = preds["boxes"]
        labels = preds["labels"]

        category_elements = {
            category: []
            for category in self.label_mapper.values()
            if category not in self.role
        }

        for box, score, label in zip(boxes, scores, labels):
            category = self.label_mapper[label.item()]

            role = None
            if category in self.role:
                role = category
                category = "paragraphs"

            category_elements[category].append(
                {
                    "box": box.astype(int).tolist(),
                    "score": float(score),
                    "role": role,
                    "order": 0,
                }
            )

        category_elements = filter_contained_rectangles_within_category(
            category_elements
        )

        return category_elements

    def __call__(self, img):
        ori_h, ori_w = img.shape[:2]
        img_tensor = self.preprocess(img)

        if self.infer_onnx:
            input = img_tensor.numpy()
            results = self.sess.run(None, {"input": input})
            preds = {
                "pred_logits": torch.tensor(results[0]).to(self.device),
                "pred_boxes": torch.tensor(results[1]).to(self.device),
            }

        else:
            with torch.inference_mode():
                img_tensor = img_tensor.to(self.device)
                preds = self.model(img_tensor)

        results = self.postprocess(preds, (ori_h, ori_w))

        tables = [Element(**table) for table in results["tables"]]
        tables = prediction_reading_order(tables, direction="top2bottom")
        tables = sorted(tables, key=lambda x: x.order)

        vis = None
        if self.visualize:
            vis = table_detector_visualizer(
                img,
                tables,
            )

        return tables, vis
