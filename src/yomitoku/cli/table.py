import argparse
import os
import time
from pathlib import Path

import torch

from ..constants import SUPPORT_INPUT_FORMAT
from ..data.functions import load_image, load_pdf
from ..table_semantic_parser import TableSemanticParser
from ..utils.logger import set_logger
from ..utils.misc import save_image
from .main import validate_encoding

logger = set_logger(__name__, "INFO")


def parse_pages(pages_str):
    pages = set()
    for part in pages_str.split(","):
        if "-" in part:
            start, end = map(int, part.split("-"))
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))
    return sorted(pages)


def collect_files(path):
    if path.is_dir():
        files = sorted(
            f
            for f in path.rglob("*")
            if f.is_file() and f.suffix[1:].lower() in SUPPORT_INPUT_FORMAT
        )
        if not files:
            logger.warning(f"No supported files found in {path}")
        return files
    return [path]


def process_single_file(file_path, args, tsp):
    logger.info(f"Processing file: {file_path}")

    if file_path.suffix.lower() == ".pdf":
        imgs = load_pdf(str(file_path), dpi=args.dpi)
    else:
        imgs = load_image(str(file_path))

    target_pages = range(1, len(imgs) + 1)
    if args.pages is not None:
        target_pages = parse_pages(args.pages)

    for page, img in enumerate(imgs):
        if (page + 1) not in target_pages:
            continue

        logger.info(f"Processing page {page + 1}...")
        start = time.time()

        semantic_info, vis_layout, vis_ocr = tsp(
            img,
            template=args.template,
            grid_only=args.grid_only,
            kv_only=args.kv_only,
        )

        # TableSemanticParser は visualize=False でも入力画像のコピーを
        # 返すため、None 判定ではなく args.vis でゲートする
        if args.vis:
            vis_path = os.path.join(
                args.outdir, f"{file_path.stem}_p{page + 1}_layout.jpg"
            )
            save_image(vis_layout, vis_path)
            logger.info(f"Output file: {vis_path}")

            vis_path = os.path.join(
                args.outdir, f"{file_path.stem}_p{page + 1}_ocr.jpg"
            )
            save_image(vis_ocr, vis_path)
            logger.info(f"Output file: {vis_path}")

        out_path = os.path.join(args.outdir, f"{file_path.stem}_p{page + 1}.json")
        if args.raw:
            semantic_info.to_json(out_path, encoding=args.encoding)
        elif args.simple:
            semantic_info.to_simple().to_json(out_path, encoding=args.encoding)
        else:
            semantic_info.to_structured().to_json(out_path, encoding=args.encoding)
        logger.info(f"Output file: {out_path}")

        elapsed = time.time() - start
        logger.info(f"Page {page + 1} done in {elapsed:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Parse tables in document images and export whole-document "
            "structured JSON (resolved key-value/grid data with cell "
            "coordinates and paragraphs)"
        )
    )
    parser.add_argument(
        "input",
        type=str,
        help="path of target image file, PDF or directory",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default="results",
        help="output directory",
    )
    parser.add_argument(
        "-v",
        "--vis",
        action="store_true",
        help="if set, visualize the result",
    )
    parser.add_argument(
        "-l",
        "--lite",
        action="store_true",
        help="if set, use lite model",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda",
        help="device to use",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help=(
            "if set, output the raw TableSemanticParserSchema JSON "
            "(normalized, template round-trip capable) instead of the "
            "resolved structured JSON"
        ),
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help=(
            "if set, output text-only structured JSON without cell "
            "coordinates and other metadata"
        ),
    )
    parser.add_argument(
        "--cell_name",
        type=str,
        default="rtdetrv2",
        help="name of table cell detector model (default: rtdetrv2)",
    )
    parser.add_argument(
        "--cell_cfg",
        type=str,
        default=None,
        help="path of table cell detector config file",
    )
    parser.add_argument(
        "--lp_name",
        type=str,
        default="rtdetrv2v2",
        help="name of layout parser model (default: rtdetrv2v2)",
    )
    parser.add_argument(
        "--lp_cfg",
        type=str,
        default=None,
        help="path of layout parser config file",
    )
    parser.add_argument(
        "--td_name",
        type=str,
        default="dbnetv2_1",
        help="name of text detector model (default: dbnetv2_1)",
    )
    parser.add_argument(
        "--td_cfg",
        type=str,
        default=None,
        help="path of text detector config file",
    )
    parser.add_argument(
        "--tr_name",
        type=str,
        default="parseq-large-v4_1",
        help="name of text recognizer model (default: parseq-large-v4_1)",
    )
    parser.add_argument(
        "--tr_cfg",
        type=str,
        default=None,
        help="path of text recognizer config file",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help="path of table template JSON to apply instead of inference",
    )
    parser.add_argument(
        "--grid_only",
        action="store_true",
        help="if set, parse only grid regions (skip key-value items)",
    )
    parser.add_argument(
        "--kv_only",
        action="store_true",
        help="if set, parse only key-value items (skip grids)",
    )
    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help="pages to process, e.g., 1,2,5-10 (default: all pages, starting from 1)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for loading PDF files (default: 200)",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help=(
            "Specifies the character encoding for the output file to be exported. "
            "If unsupported characters are included, they will be ignored."
        ),
    )
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {args.input}")

    if args.template is not None and not os.path.exists(args.template):
        raise FileNotFoundError(f"Template file not found: {args.template}")

    if args.raw and args.simple:
        raise ValueError("--raw and --simple cannot be used together")

    validate_encoding(args.encoding)

    configs = {
        "table_detector": {
            "model_name": args.lp_name,
            "path_cfg": args.lp_cfg,
        },
        "table_cell_parser": {
            "model_name": args.cell_name,
            "path_cfg": args.cell_cfg,
        },
        "text_detector": {
            "model_name": args.td_name,
            "path_cfg": args.td_cfg,
        },
        "text_recognizer": {
            "model_name": args.tr_name,
            "path_cfg": args.tr_cfg,
        },
    }

    if args.lite:
        # cli/main.py の --lite と同等の軽量化。明示的に --tr_name が
        # 指定された場合はそちらを優先する
        lite_tr_name = "parseq-tiny"
        if args.tr_name != "parseq-large-v4_1":
            lite_tr_name = args.tr_name
        configs["text_recognizer"]["model_name"] = lite_tr_name

        if args.device == "cpu" or not torch.cuda.is_available():
            configs["text_detector"]["infer_onnx"] = True

    tsp = TableSemanticParser(
        configs=configs,
        device=args.device,
        visualize=args.vis,
    )

    os.makedirs(args.outdir, exist_ok=True)
    logger.info(f"Output directory: {args.outdir}")

    files = collect_files(path)
    for file_path in files:
        try:
            start = time.time()
            process_single_file(file_path, args, tsp)
            elapsed = time.time() - start
            logger.info(f"Total Processing time: {elapsed:.2f} sec")
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            continue


if __name__ == "__main__":
    main()
