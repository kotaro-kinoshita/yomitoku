import argparse
import os
import time
from pathlib import Path

from ..constants import SUPPORT_INPUT_FORMAT
from ..data.functions import load_image, load_pdf
from ..extractor.rule_pipeline import run_rule_extraction
from ..extractor.schema import ExtractionSchema
from ..table_semantic_parser import TableSemanticParser
from ..utils.logger import set_logger
from ..utils.misc import save_image

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


def process_single_file(file_path, args, tsp, schema):
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

        semantic_info, vis_layout, vis_ocr = tsp(img)

        if args.vis and vis_layout is not None:
            vis_path = os.path.join(
                args.outdir, f"{file_path.stem}_p{page + 1}_layout.jpg"
            )
            save_image(vis_layout, vis_path)

        if args.vis and vis_ocr is not None:
            vis_path = os.path.join(
                args.outdir, f"{file_path.stem}_p{page + 1}_ocr.jpg"
            )
            save_image(vis_ocr, vis_path)

        filename = f"{file_path.stem}_p{page + 1}_extract"
        run_rule_extraction(
            semantic_info=semantic_info,
            img=img,
            schema=schema,
            no_normalize=args.no_normalize,
            visualize=args.vis,
            simple=args.simple,
            outdir=args.outdir,
            filename=filename,
        )

        elapsed = time.time() - start
        logger.info(f"Page {page + 1} done in {elapsed:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Extract structured data from document images using rule-based matching"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input image, PDF path, or directory",
    )
    parser.add_argument(
        "-s",
        "--schema",
        type=str,
        required=True,
        help="Extraction schema file (YAML)",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default="results",
        help="Output directory (default: results)",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda",
        help="Device for TableSemanticParser (default: cuda)",
    )
    parser.add_argument(
        "-v",
        "--vis",
        action="store_true",
        help="Output visualization images",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip text normalization",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Output simple {name: value} format without bbox/metadata",
    )
    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help="Pages to process, e.g. 1,2,5-10 (default: all)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for PDF loading (default: 200)",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="Output encoding (default: utf-8)",
    )

    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {args.input}")

    schema_path = Path(args.schema)
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {args.schema}")

    schema = ExtractionSchema.from_yaml(str(schema_path))
    logger.info(f"Loaded schema with {len(schema.fields)} fields")

    tsp = TableSemanticParser(
        configs={},
        device=args.device,
        visualize=args.vis,
    )

    os.makedirs(args.outdir, exist_ok=True)

    files = collect_files(path)
    for file_path in files:
        try:
            process_single_file(file_path, args, tsp, schema)
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            continue


if __name__ == "__main__":
    main()
