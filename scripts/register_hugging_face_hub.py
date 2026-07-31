import argparse
import torch

from yomitoku.layout_parser import LayoutParser
from yomitoku.table_cell_detector import CellDetector
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

    elif module_name == "table_cell_detector":
        kwargs = dict(from_pretrained=False, device=device)
        if model_name:
            kwargs["model_name"] = model_name
        module = CellDetector(**kwargs)
        return module

    raise ValueError(f"Invalid module name: {module_name}")


def load_state_dict(checkpoint_path, weights_key):
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if weights_key == "ema":
        # rtdetrv2_pytorch の学習チェックポイントは EMA 重みを
        # ckpt["ema"]["module"] に持つ
        return ckpt["ema"]["module"]

    return ckpt[weights_key]


def main(args):
    module = get_module(args.module, args.model_name, args.device)

    state_dict = load_state_dict(args.checkpoint, args.weights_key)
    # strict=True: missing/unexpected keys があれば例外で中断し、
    # 不完全な重みが保存・アップロードされるのを防ぐ
    module.model.load_state_dict(state_dict, strict=True)

    module.model.save_pretrained(args.name)
    print(f"Saved HF-format model to ./{args.name}")

    if args.push:
        repo_id = f"{args.owner}/{args.name}"
        module.model.push_to_hub(repo_id, token=args.token)
        print(f"Pushed to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=str)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument(
        "--weights_key",
        type=str,
        default="model",
        help="checkpoint key to load weights from ('model' or 'ema')",
    )
    parser.add_argument("--owner", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument(
        "--push",
        action="store_true",
        help="if set, push the converted model to HF Hub",
    )
    args = parser.parse_args()

    main(args)
