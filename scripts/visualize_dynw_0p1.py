"""Visualize and time DocumentAnalyzer with the 0+1 speedup config.

Runs the full DocumentAnalyzer pipeline on a directory of pages using the
dynw fine-tuned recognizer with batch_bucketing + dynamic_width (the "0+1"
combo), saves OCR and layout visualizations, and reports the average
per-page wall-clock time (first page treated as warmup).

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/visualize_dynw_0p1.py \
        --input dataset/test_20241014_good/0000 --limit 20 --out results_dynw_0p1
"""

import argparse
import time
from pathlib import Path

import cv2
import torch

from yomitoku import DocumentAnalyzer
from yomitoku.data.functions import load_image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="results_dynw_0p1")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--model", default="parseq-large-v4_1-dynw")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        f for f in Path(args.input).iterdir() if f.suffix.lower() in IMAGE_EXTS
    )[: args.limit]
    print(f"device={device}  pages={len(files)}  out={out_dir}")

    configs = {
        "ocr": {
            "text_recognizer": {
                "model_name": args.model,
                "batch_bucketing": True,  # stage 0
                "dynamic_width": True,  # stage 1
            },
        },
    }
    analyzer = DocumentAnalyzer(configs=configs, device=device, visualize=True)

    per_page = []
    for i, f in enumerate(files):
        img = load_image(str(f))[0]
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        results, ocr_vis, layout_vis = analyzer(img)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        cv2.imwrite(str(out_dir / f"{f.stem}_ocr.jpg"), ocr_vis)
        cv2.imwrite(str(out_dir / f"{f.stem}_layout.jpg"), layout_vis)

        tag = " (warmup, excluded)" if i == 0 else ""
        print(f"[{i:02d}] {f.name}: {elapsed:6.3f}s{tag}")
        if i > 0:
            per_page.append(elapsed)

    if per_page:
        mean = sum(per_page) / len(per_page)
        print(
            f"\n=== per-page average over {len(per_page)} pages "
            f"(warmup excluded): {mean:.3f}s/page ===\n"
            f"min={min(per_page):.3f}s  max={max(per_page):.3f}s"
        )


if __name__ == "__main__":
    main()
