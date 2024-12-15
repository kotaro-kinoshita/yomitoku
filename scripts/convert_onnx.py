import argparse

import os
import torch

from yomitoku.layout_parser import LayoutParser
from yomitoku.table_structure_recognizer import TableStructureRecognizer
from yomitoku.text_detector import TextDetector
from yomitoku.text_recognizer import TextRecognizer


def convert_onnx(name, dir, model, img_size):
    model.eval()

    dynamic_axes = {
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    }

    if img_size is None:
        dynamic_axes["input"] = {0: "batch_size", 2: "height", 3: "width"}
        dynamic_axes["output"] = {0: "batch_size", 2: "height", 3: "width"}

    if img_size is None:
        img_size = (256, 256)

    os.makedirs(dir, exist_ok=True)
    output_path = os.path.join(dir, f"{name}.onnx")

    dummy_input = torch.randn(1, 3, *img_size, requires_grad=True).cuda()

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=14,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )


def get_module(module_name):
    if module_name == "text_detector":
        module = TextDetector()
        return module

    elif module_name == "text_recognizer":
        module = TextRecognizer()
        return module

    elif module_name == "layout_parser":
        module = LayoutParser()
        return module

    elif module_name == "table_structure_recognizer":
        module = TableStructureRecognizer()
        return module

    raise ValueError(f"Invalid module name: {module_name}")


def main(args):
    module = get_module(args.module)
    img_size = None
    name = os.path.basename(module._cfg.hf_hub_repo)
    model = module.model

    if hasattr(module._cfg, "data") and hasattr(module._cfg.data, "img_size"):
        img_size = module._cfg.data.img_size

    convert_onnx(name, args.dir, model, img_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=str)
    parser.add_argument("--dir", type=str)
    args = parser.parse_args()

    main(args)
