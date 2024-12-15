import argparse
import os
from pathlib import Path

import cv2
import time

from ..constants import SUPPORT_OUTPUT_FORMAT
from ..data.functions import load_image, load_pdf
from ..document_analyzer import DocumentAnalyzer
from ..utils.logger import set_logger

logger = set_logger(__name__, "INFO")


def process_single_file(args, analyzer, path, format):
    if path.suffix[1:].lower() in ["pdf"]:
        imgs = load_pdf(path)
    else:
        imgs = [load_image(path)]

    for page, img in enumerate(imgs):
        results, ocr, layout = analyzer(img)

        dirname = path.parent.name
        filename = path.stem

        if ocr is not None:
            out_path = os.path.join(
                args.outdir, f"{dirname}_{filename}_p{page+1}_ocr.jpg"
            )

            cv2.imwrite(out_path, ocr)
            logger.info(f"Output file: {out_path}")

        if layout is not None:
            out_path = os.path.join(
                args.outdir, f"{dirname}_{filename}_p{page+1}_layout.jpg"
            )

            cv2.imwrite(out_path, layout)
            logger.info(f"Output file: {out_path}")

        out_path = os.path.join(args.outdir, f"{dirname}_{filename}_p{page+1}.{format}")

        if format == "json":
            results.to_json(
                out_path,
                ignore_line_break=args.ignore_line_break,
            )
        elif format == "csv":
            results.to_csv(
                out_path,
                ignore_line_break=args.ignore_line_break,
            )
        elif format == "html":
            results.to_html(
                out_path,
                ignore_line_break=args.ignore_line_break,
                img=img,
                export_figure=args.figure,
                export_figure_letter=args.figure_letter,
                figure_width=args.figure_width,
                figure_dir=args.figure_dir,
            )
        elif format == "md":
            results.to_markdown(
                out_path,
                ignore_line_break=args.ignore_line_break,
                img=img,
                export_figure=args.figure,
                export_figure_letter=args.figure_letter,
                figure_width=args.figure_width,
                figure_dir=args.figure_dir,
            )

        logger.info(f"Output file: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "arg1",
        type=str,
        help="path of target image file or directory",
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        default="json",
        help="output format type (json or csv or html or md)",
    )
    parser.add_argument(
        "-v",
        "--vis",
        action="store_true",
        help="if set, visualize the result",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default="results",
        help="output directory",
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
    parser.add_argument(
        "--ignore_line_break",
        action="store_true",
        help="if set, ignore line break in the output",
    )
    parser.add_argument(
        "--figure",
        action="store_true",
        help="if set, export figure in the output",
    )
    parser.add_argument(
        "--figure_letter",
        action="store_true",
        help="if set, export letter within figure in the output",
    )
    parser.add_argument(
        "--figure_width",
        type=int,
        default=200,
        help="width of figure image in the output",
    )
    parser.add_argument(
        "--figure_dir",
        type=str,
        default="figures",
        help="directory to save figure images",
    )

    args = parser.parse_args()

    path = Path(args.arg1)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {args.arg1}")

    format = args.format.lower()
    if format not in SUPPORT_OUTPUT_FORMAT:
        raise ValueError(
            f"Invalid output format: {args.format}. Supported formats are {SUPPORT_OUTPUT_FORMAT}"
        )

    if format == "markdown":
        format = "md"

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

    if args.lite:
        configs["ocr"]["text_recognizer"]["model_name"] = "parseq-small"

    analyzer = DocumentAnalyzer(
        configs=configs,
        visualize=args.vis,
        device=args.device,
    )

    os.makedirs(args.outdir, exist_ok=True)
    logger.info(f"Output directory: {args.outdir}")

    if path.is_dir():
        all_files = [f for f in path.rglob("*") if f.is_file()]
        for f in all_files:
            try:
                start = time.time()
                file_path = Path(f)
                logger.info(f"Processing file: {file_path}")
                process_single_file(args, analyzer, file_path, format)
                end = time.time()
                logger.info(f"Total Processing time: {end-start:.2f} sec")
            except Exception:
                continue
    else:
        start = time.time()
        logger.info(f"Processing file: {path}")
        process_single_file(args, analyzer, path, format)
        end = time.time()
        logger.info(f"Total Processing time: {end-start:.2f} sec")


if __name__ == "__main__":
    main()
