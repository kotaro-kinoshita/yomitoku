import argparse


from yomitoku.layout_parser import LayoutParser
from yomitoku.table_structure_recognizer import TableStructureRecognizer
from yomitoku.text_detector import TextDetector
from yomitoku.text_recognizer import TextRecognizer


def get_module(module_name, device):
    if module_name == "text_detector":
        module = TextDetector(
            from_pretrained=True,
            device=device,
        )
        return module

    elif module_name == "text_recognizer":
        module = TextRecognizer(
            from_pretrained=True,
            device=device,
        )
        return module

    elif module_name == "layout_parser":
        module = LayoutParser(
            from_pretrained=True,
            device=device,
        )
        return module

    elif module_name == "table_structure_recognizer":
        module = TableStructureRecognizer(
            from_pretrained=False,
            device=device,
        )
        return module

    raise ValueError(f"Invalid module name: {module_name}")


def main(args):
    module = get_module(args.module, args.device)
    # module.model.load_state_dict(
    #    torch.load(args.checkpoint, map_location="cpu")["model"]
    # )

    module.model.save_pretrained(args.name)
    module.model.push_to_hub(f"{args.owner}/{args.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--owner", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    main(args)
