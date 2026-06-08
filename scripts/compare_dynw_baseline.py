"""Compare DocumentAnalyzer baseline vs the 0+1 speedup config.

Times the full DocumentAnalyzer pipeline on the same pages with the dynw
recognizer in two modes -- baseline (no bucketing, no dynamic width) and
0+1 (batch_bucketing + dynamic_width) -- and reports the average per-page
wall-clock time, speedup, and how often the extracted text agrees.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/compare_dynw_baseline.py \
        --input dataset/test_20241014_good/0000 --limit 20
"""

import argparse
import time
from pathlib import Path

import torch

from yomitoku import DocumentAnalyzer
from yomitoku.data.functions import load_image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

CONFIGS = [
    ("baseline", dict(batch_bucketing=False, dynamic_width=False)),
    ("0+1 (bucket+dynw)", dict(batch_bucketing=True, dynamic_width=True)),
]


def page_text(results):
    return "\n".join(p.contents for p in results.paragraphs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--model", default="parseq-large-v4_1-dynw")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    files = sorted(
        f for f in Path(args.input).iterdir() if f.suffix.lower() in IMAGE_EXTS
    )[: args.limit]
    imgs = [(f.name, load_image(str(f))[0]) for f in files]
    print(f"device={device}  pages={len(imgs)}\n")

    analyzer = DocumentAnalyzer(
        configs={"ocr": {"text_recognizer": {"model_name": args.model}}},
        device=device,
        visualize=False,
    )

    # Warmup (excluded from all timings).
    analyzer(imgs[0][1])

    texts = {}
    per_page = {}
    for name, flags in CONFIGS:
        analyzer.text_recognizer.batch_bucketing = flags["batch_bucketing"]
        analyzer.text_recognizer.dynamic_width = flags["dynamic_width"]

        times = []
        outs = []
        for _, img in imgs:
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            results, _, _ = analyzer(img)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            outs.append(page_text(results))
        per_page[name] = times
        texts[name] = outs
        mean = sum(times) / len(times)
        print(f"[{name:18s}] avg={mean:.3f}s/page  total={sum(times):.2f}s")

    base = "baseline"
    print("\n=== summary (per-page average) ===")
    print(f"{'config':<18s} {'avg[s/page]':>12s} {'speedup':>8s} {'text-match':>11s}")
    base_mean = sum(per_page[base]) / len(per_page[base])
    for name, _ in CONFIGS:
        mean = sum(per_page[name]) / len(per_page[name])
        match = sum(
            1 for a, b in zip(texts[base], texts[name]) if a == b
        ) / len(texts[base])
        print(f"{name:<18s} {mean:12.3f} {base_mean / mean:7.2f}x {match:10.1%}")


if __name__ == "__main__":
    main()
