"""Benchmark for the TextRecognizer batch_bucketing option.

Measures the TextRecognizer end-to-end latency (preprocess + inference +
postprocess) on the same images and the same detected polygons with and
without width bucketing, on CPU and CUDA.

Usage:
    python scripts/benchmark_parseq_speedup.py --input static/in --devices cuda cpu
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from yomitoku import TextDetector, TextRecognizer
from yomitoku.data.dataset import ParseqDataset
from yomitoku.data.functions import load_image, load_pdf

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".pdf"}

CONFIGS = [
    ("baseline", dict(batch_bucketing=False)),
    ("bucketing", dict(batch_bucketing=True)),
]


def expand_inputs(paths):
    files = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files.extend(
                sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
            )
        else:
            files.append(p)
    return files


def load_target(path):
    path = str(path)
    if path.lower().endswith(".pdf"):
        return load_pdf(path)[0]
    return load_image(path)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", default=["static/in"])
    parser.add_argument("--model", default="parseq-large-v4_1")
    parser.add_argument("--devices", nargs="+", default=["cuda", "cpu"])
    parser.add_argument("--warmup", type=int, default=1, help="warmup passes (cuda)")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--cpu-warmup", type=int, default=0)
    parser.add_argument("--cpu-runs", type=int, default=1)
    args = parser.parse_args()

    files = expand_inputs(args.input)
    det_device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = TextDetector(device=det_device, visualize=False)

    workload = []  # (name, img, points)
    rec_cfg = TextRecognizer(model_name=args.model, device=det_device)._cfg
    total_lines = 0
    for f in files:
        img = load_target(f)
        det_results, _ = detector(img)
        points = det_results.points
        if len(points) == 0:
            print(f"{f.name}: no lines, skipped")
            continue
        ds = ParseqDataset(rec_cfg, img, points)
        w = np.array(ds.content_widths)
        print(
            f"{f.name}: lines={len(points)} width med={np.median(w):.0f} "
            f"max={w.max()} fill={np.mean(w) / rec_cfg.data.img_size[1]:5.1%}"
        )
        workload.append((f.name, img, points))
        total_lines += len(points)
    print(f"total: {len(workload)} images, {total_lines} lines\n")

    all_rows = []
    for device in args.devices:
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, skipping")
            continue
        warmup = args.warmup if device == "cuda" else args.cpu_warmup
        runs = args.runs if device == "cuda" else args.cpu_runs

        recognizer = TextRecognizer(
            model_name=args.model, device=device, visualize=False
        )

        base_preds = None
        for name, flags in CONFIGS:
            recognizer.batch_bucketing = flags["batch_bucketing"]

            pass_times = []
            preds = None
            for i in range(warmup + runs):
                preds = []
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _, img, points in workload:
                    results, _ = recognizer(img, points)
                    preds.extend(results.contents)
                if device == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                if i >= warmup:
                    pass_times.append(elapsed)

            if base_preds is None:
                base_preds = preds
                rate = 1.0
            else:
                rate = sum(1 for b, o in zip(base_preds, preds) if b == o) / len(
                    base_preds
                )

            mean = sum(pass_times) / len(pass_times)
            all_rows.append((device, name, mean, min(pass_times), rate))
            print(
                f"[{device}] {name:<10s} mean={mean:8.3f}s min={min(pass_times):8.3f}s "
                f"exact-match vs baseline={rate:6.1%}"
            )

        del recognizer
        if device == "cuda":
            torch.cuda.empty_cache()

    print("\n=== summary ===")
    print(
        f"{'device':<6s} {'config':<10s} {'mean[s]':>9s} {'min[s]':>9s} "
        f"{'speedup':>8s} {'match':>7s}"
    )
    base_mean = {}
    for device, name, mean, tmin, rate in all_rows:
        if name == "baseline":
            base_mean[device] = mean
        speedup = base_mean[device] / mean
        print(
            f"{device:<6s} {name:<10s} {mean:9.3f} {tmin:9.3f} {speedup:7.2f}x {rate:6.1%}"
        )


if __name__ == "__main__":
    main()
