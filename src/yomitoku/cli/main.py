import argparse
import os
from pathlib import Path

import cv2

from ..data.functions import load_image
from ..document_analyzer import DocumentAnalyzer


def process_single_file(args, analyzer, path, format):
    img = load_image(path)
    results, ocr, layout = analyzer(img)

    dirname = path.parent.name
    filename = path.stem

    if ocr is not None:
        cv2.imwrite(
            os.path.join(args.outdir, f"{dirname}_{filename}_ocr.jpg"),
            ocr,
        )

    if layout is not None:
        cv2.imwrite(
            os.path.join(args.outdir, f"{dirname}_{filename}_layout.jpg"),
            layout,
        )

    out_path = os.path.join(args.outdir, f"{dirname}_{filename}.{format}")

    if args.format == "json":
        results.to_json(out_path)
    elif args.format == "csv":
        results.to_csv(out_path)
    elif args.format == "html":
        results.to_html(out_path)
    elif args.format == "md":
        results.to_markdown(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "arg1", type=str, help="path of target image file or directory"
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        default="json",
        help="output format type (json or csv or html or md)",
    )
    parser.add_argument(
        "-v", "--vis", action="store_true", help="if set, visualize the result"
    )
    parser.add_argument(
        "-o", "--outdir", type=str, default="results", help="output directory"
    )
    parser.add_argument(
        "-d", "--device", type=str, default="cuda", help="device to use"
    )
    parser.add_argument(
        "--td_cfg",
        type=str,
        default=None,
        help="path of text detector config file",
    )
    parser.add_argument(
        "--tr_cfg",
        type=str,
        default=None,
        help="path of text recognizer config file",
    )
    parser.add_argument(
        "--lp_cfg",
        type=str,
        default=None,
        help="path of layout parser config file",
    )
    parser.add_argument(
        "--tsr_cfg",
        type=str,
        default=None,
        help="path of table structure recognizer config file",
    )
    args = parser.parse_args()

    try:
        path = Path(args.arg1)
    except Exception:
        raise ValueError(f"Invalid path: {args.arg1}")

    if not path.exists():
        raise FileNotFoundError(f"File not found: {args.arg1}")

    suport_formats = ["json", "csv", "html", "md"]
    if args.format not in suport_formats:
        raise ValueError(f"Invalid output format: {args.format}")

    configs = {
        "ocr": {
            "text_detector": {
                "path_cfg": args.td_cfg,
            },
            "text_recognizer": {
                "path_cfg": args.tr_cfg,
            },
        },
        "layout_analyzer": {
            "layout_parser": {
                "path_cfg": args.lp_cfg,
            },
            "table_structure_recognizer": {
                "path_cfg": args.tsr_cfg,
            },
        },
    }

    analyzer = DocumentAnalyzer(
        configs=configs,
        visualize=args.vis,
        device=args.device,
    )

    os.makedirs(args.outdir, exist_ok=True)
    if path.is_dir():
        all_files = [f for f in path.rglob("*") if f.is_file()]
        for f in all_files:
            try:
                file_path = Path(f)
                process_single_file(args, analyzer, file_path, args.format)
            except Exception:
                continue
    else:
        process_single_file(args, analyzer, path, args.format)


if __name__ == "__main__":
    main()
