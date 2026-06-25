import argparse
import torch

from yomitoku.layout_parser import LayoutParser
from yomitoku.table_structure_recognizer import TableStructureRecognizer
from yomitoku.text_detector import TextDetector
from yomitoku.text_recognizer import TextRecognizer


def get_module(module_name, model_name, device):
    if module_name == "text_detector":
        kwargs = dict(from_pretrained=False, device=device)
        if model_name:
            kwargs["model_name"] = model_name
        module = TextDetector(**kwargs)
        return module

    elif module_name == "text_recognizer":
        kwargs = dict(from_pretrained=False, device=device)
        if model_name:
            kwargs["model_name"] = model_name
        module = TextRecognizer(**kwargs)
        return module

    elif module_name == "layout_parser":
        kwargs = dict(from_pretrained=False, device=device)
        if model_name:
            kwargs["model_name"] = model_name
        module = LayoutParser(**kwargs)
        return module

    elif module_name == "table_structure_recognizer":
        kwargs = dict(from_pretrained=False, device=device)
        if model_name:
            kwargs["model_name"] = model_name
        module = TableStructureRecognizer(**kwargs)
        return module

    raise ValueError(f"Invalid module name: {module_name}")


def main(args):
    module = get_module(args.module, args.model_name, args.device)
    module.model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu")["model"]
    )

    module.model.save_pretrained(args.name)
    module.model.push_to_hub(f"{args.owner}/{args.name}", token=args.token)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=str)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--owner", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--token", type=str)
    args = parser.parse_args()

    main(args)
