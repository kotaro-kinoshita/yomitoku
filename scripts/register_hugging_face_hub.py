import argparse

import torch

from yomitoku.layout_parse import LayoutParser
from yomitoku.table_structure_recognition import TableStructureRecognizer
from yomitoku.text_detection import TextDetector
from yomitoku.text_recognition import TextRecognizer


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
    module.model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu")["model"]
    )

    module.model.save_pretrained(args.name)
    module.model.push_to_hub(f"{args.owner}/{args.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--owner", type=str)
    parser.add_argument("--name", type=str)
    args = parser.parse_args()

    main(args)
