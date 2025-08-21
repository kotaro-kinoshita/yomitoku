import argparse
import os
import re
import time
from pathlib import Path

import torch

from ..constants import SUPPORT_OUTPUT_FORMAT
from ..data.functions import load_image, load_pdf
from ..document_analyzer import DocumentAnalyzer
from ..utils.logger import set_logger
from ..utils.searchable_pdf import create_searchable_pdf

from ..export import save_csv, save_html, save_json, save_markdown
from ..export import convert_json, convert_csv, convert_html, convert_markdown

from ..utils.misc import save_image

logger = set_logger(__name__, "INFO")


def merge_all_pages(results):
    out = None
    for result in results:
        format = result["format"]
        data = result["data"]

        if format == "json":
            if out is None:
                out = [data]
            else:
                out.append(data)

        elif format == "csv":
            if out is None:
                out = data
            else:
                out.extend(data)

        elif format == "html":
            if out is None:
                out = data
            else:
                out += "\n" + data

        elif format == "md":
            if out is None:
                out = data
            else:
                out += "\n" + data

        elif format == "pdf":
            if out is None:
                out = [data]
            else:
                out.append(data)
    return out


def save_merged_file(out_path, args, out, imgs):
    if args.format == "json":
        save_json(out, out_path, args.encoding)
    elif args.format == "csv":
        save_csv(out, out_path, args.encoding)
    elif args.format == "html":
        save_html(out, out_path, args.encoding)
    elif args.format == "md":
        save_markdown(out, out_path, args.encoding)
    elif args.format == "pdf":
        create_searchable_pdf(
            imgs,
            out,
            output_path=out_path,
            font_path=args.font_path,
        )


def validate_encoding(encoding):
    if encoding not in [
        "utf-8",
        "utf-8-sig",
        "shift-jis",
        "euc-jp",
        "cp932",
    ]:
        raise ValueError(f"Invalid encoding: {encoding}")
    return True


def process_single_file(args, analyzer, path, format):
    if path.suffix[1:].lower() in ["pdf"]:
        imgs = load_pdf(path)
    else:
        imgs = load_image(path)

    format_results = []
    for page, img in enumerate(imgs):
        result, ocr, layout = analyzer(img)
        dirname = _sanitize_path_component(path.parent.name)
        filename = path.stem

        # cv2.imwrite(
        #    os.path.join(args.outdir, f"{dirname}_{filename}_p{page+1}.jpg"), img
        # )

        if ocr is not None:
            out_path = os.path.join(
                args.outdir, f"{dirname}_{filename}_p{page + 1}_ocr.jpg"
            )

            save_image(ocr, out_path)
            logger.info(f"Output file: {out_path}")

        if layout is not None:
            out_path = os.path.join(
                args.outdir, f"{dirname}_{filename}_p{page + 1}_layout.jpg"
            )

            save_image(layout, out_path)
            logger.info(f"Output file: {out_path}")

        out_path = os.path.join(
            args.outdir, f"{dirname}_{filename}_p{page + 1}.{format}"
        )

        if format == "json":
            if args.combine:
                json = convert_json(
                    result,
                    out_path,
                    args.ignore_line_break,
                    img,
                    args.figure,
                    args.figure_dir,
                )
            else:
                json = result.to_json(
                    out_path,
                    ignore_line_break=args.ignore_line_break,
                    encoding=args.encoding,
                    img=img,
                    export_figure=args.figure,
                    figure_dir=args.figure_dir,
                )

            format_results.append(
                {
                    "format": format,
                    "data": json.model_dump(),
                }
            )

        elif format == "csv":
            if args.combine:
                csv = convert_csv(
                    result,
                    out_path,
                    args.ignore_line_break,
                    img,
                    args.figure,
                    args.figure_letter,
                    args.figure_dir,
                )
            else:
                csv = result.to_csv(
                    out_path,
                    ignore_line_break=args.ignore_line_break,
                    encoding=args.encoding,
                    img=img,
                    export_figure=args.figure,
                    export_figure_letter=args.figure_letter,
                    figure_dir=args.figure_dir,
                )

            format_results.append(
                {
                    "format": format,
                    "data": csv,
                }
            )

        elif format == "html":
            if args.combine:
                html, _ = convert_html(
                    result,
                    out_path,
                    ignore_line_break=args.ignore_line_break,
                    img=img,
                    export_figure=args.figure,
                    export_figure_letter=args.figure_letter,
                    figure_width=args.figure_width,
                    figure_dir=args.figure_dir,
                )
            else:
                html = result.to_html(
                    out_path,
                    ignore_line_break=args.ignore_line_break,
                    img=img,
                    export_figure=args.figure,
                    export_figure_letter=args.figure_letter,
                    figure_width=args.figure_width,
                    figure_dir=args.figure_dir,
                    encoding=args.encoding,
                )

            format_results.append(
                {
                    "format": format,
                    "data": html,
                }
            )

        elif format == "md":
            if args.combine:
                md, _ = convert_markdown(
                    result,
                    out_path,
                    ignore_line_break=args.ignore_line_break,
                    img=img,
                    export_figure=args.figure,
                    export_figure_letter=args.figure_letter,
                    figure_width=args.figure_width,
                    figure_dir=args.figure_dir,
                )
            else:
                md = result.to_markdown(
                    out_path,
                    ignore_line_break=args.ignore_line_break,
                    img=img,
                    export_figure=args.figure,
                    export_figure_letter=args.figure_letter,
                    figure_width=args.figure_width,
                    figure_dir=args.figure_dir,
                    encoding=args.encoding,
                )

            format_results.append(
                {
                    "format": format,
                    "data": md,
                }
            )
        elif format == "pdf":
            if not args.combine:
                create_searchable_pdf(
                    [img],
                    [result],
                    output_path=out_path,
                    font_path=args.font_path,
                )

            format_results.append(
                {
                    "format": format,
                    "data": result,
                }
            )

    out = merge_all_pages(format_results)
    if args.combine:
        out_path = os.path.join(args.outdir, f"{dirname}_{filename}.{format}")
        save_merged_file(
            out_path,
            args,
            out,
            imgs,
        )


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
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="Specifies the character encoding for the output file to be exported. If unsupported characters are included, they will be ignored.",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="if set, merge all pages in the output",
    )
    parser.add_argument(
        "--ignore_meta",
        action="store_true",
        help="if set, ignore meta information(header, footer) in the output",
    )
    parser.add_argument(
        "--reading_order",
        default="auto",
        type=str,
        choices=["auto", "left2right", "top2bottom", "right2left"],
    )
    parser.add_argument(
        "--font_path",
        default=None,
        type=str,
        help="Path to the font file(.ttf) for PDF output",
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

    if (
        args.font_path is not None
        and not os.path.exists(args.font_path)
        and format == "pdf"
    ):
        raise FileNotFoundError(f"Font file not found: {args.font_path}")

    validate_encoding(args.encoding)

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

        if args.device == "cpu" or not torch.cuda.is_available():
            configs["ocr"]["text_detector"]["infer_onnx"] = True

        # Note: Text Detector以外はONNX推論よりもPyTorch推論の方が速いため、ONNX推論は行わない
        # configs["ocr"]["text_recognizer"]["infer_onnx"] = True
        # configs["layout_analyzer"]["table_structure_recognizer"]["infer_onnx"] = True
        # configs["layout_analyzer"]["layout_parser"]["infer_onnx"] = True

    analyzer = DocumentAnalyzer(
        configs=configs,
        visualize=args.vis,
        device=args.device,
        ignore_meta=args.ignore_meta,
        reading_order=args.reading_order,
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
                logger.info(f"Total Processing time: {end - start:.2f} sec")
            except Exception:
                continue
    else:
        start = time.time()
        logger.info(f"Processing file: {path}")
        process_single_file(args, analyzer, path, format)
        end = time.time()
        logger.info(f"Total Processing time: {end - start:.2f} sec")


def _sanitize_path_component(component):
    if not component:
        return component

    return re.sub(r"^\.+", lambda m: "_" * len(m.group(0)), component)


if __name__ == "__main__":
    main()
