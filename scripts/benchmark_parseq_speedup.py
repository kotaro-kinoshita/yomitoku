"""Benchmark for TextRecognizer batch_bucketing / dynamic_width options.

Measures the TextRecognizer end-to-end latency (preprocess + inference +
postprocess) on the same image and the same detected polygons for the four
combinations of the two options, on CPU and CUDA.

Usage:
    python scripts/benchmark_parseq_speedup.py [--image PATH] [--devices cuda cpu]
"""

import argparse
import time

import numpy as np
import torch

from yomitoku import TextDetector, TextRecognizer
from yomitoku.data.functions import load_image, load_pdf

CONFIGS = [
    ("baseline", dict(batch_bucketing=False, dynamic_width=False)),
    ("bucketing (stage0)", dict(batch_bucketing=True, dynamic_width=False)),
    ("dynamic_width (stage1)", dict(batch_bucketing=False, dynamic_width=True)),
    ("bucketing+dynamic (0+1)", dict(batch_bucketing=True, dynamic_width=True)),
]


def load_target(path):
    if path.lower().endswith(".pdf"):
        return load_pdf(path)[0]
    return load_image(path)


def tile_workload(img, points, n):
    """Tile the image n x n and replicate the polygons with offsets, to
    scale the number of text lines beyond one mini-batch."""
    if n <= 1:
        return img, points
    h, w = img.shape[:2]
    tiled = np.tile(img, (n, n, 1))
    tiled_points = []
    for ty in range(n):
        for tx in range(n):
            offset = np.array([tx * w, ty * h])
            for quad in points:
                tiled_points.append((np.array(quad) + offset).tolist())
    return tiled, tiled_points


def run_config(recognizer, img, points, warmup, runs, device):
    times = []
    results = None
    for i in range(warmup + runs):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        results, _ = recognizer(img, points)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        if i >= warmup:
            times.append(elapsed)
    return times, results


def match_rate(base, other):
    assert len(base) == len(other)
    n_match = sum(1 for b, o in zip(base, other) if b == o)
    return n_match / len(base)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="demo/sample.pdf")
    parser.add_argument("--model", default="parseq-large-v4_1")
    parser.add_argument("--devices", nargs="+", default=["cuda", "cpu"])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--cpu-runs", type=int, default=2)
    parser.add_argument("--tile", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    img = load_target(args.image)
    print(f"image: {args.image} shape={img.shape}")

    detector = TextDetector(
        device="cuda" if torch.cuda.is_available() else "cpu", visualize=False
    )
    det_results, _ = detector(img)
    points = det_results.points
    img, points = tile_workload(img, points, args.tile)
    print(f"lines: {len(points)} (tile={args.tile})")

    all_rows = []
    for device in args.devices:
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, skipping")
            continue
        runs = args.cpu_runs if device == "cpu" else args.runs

        recognizer = TextRecognizer(
            model_name=args.model, device=device, visualize=False
        )
        if args.batch_size is not None:
            recognizer._cfg.data.batch_size = args.batch_size
        print(f"[{device}] batch_size={recognizer._cfg.data.batch_size}")

        base_preds = None
        for name, flags in CONFIGS:
            recognizer.batch_bucketing = flags["batch_bucketing"]
            recognizer.dynamic_width = flags["dynamic_width"]

            times, results = run_config(
                recognizer, img, points, args.warmup, runs, device
            )
            preds = list(results.contents)
            if base_preds is None:
                base_preds = preds
                rate = 1.0
            else:
                rate = match_rate(base_preds, preds)

            mean = sum(times) / len(times)
            row = (device, name, mean, min(times), rate)
            all_rows.append(row)
            print(
                f"[{device}] {name:<26s} mean={mean:7.3f}s min={min(times):7.3f}s "
                f"exact-match vs baseline={rate:6.1%}"
            )

        del recognizer
        if device == "cuda":
            torch.cuda.empty_cache()

    print("\n=== summary ===")
    print(
        f"{'device':<6s} {'config':<26s} {'mean[s]':>9s} {'min[s]':>9s} {'speedup':>8s} {'match':>7s}"
    )
    base_mean = {}
    for device, name, mean, tmin, rate in all_rows:
        if name == "baseline":
            base_mean[device] = mean
        speedup = base_mean[device] / mean
        print(
            f"{device:<6s} {name:<26s} {mean:9.3f} {tmin:9.3f} {speedup:7.2f}x {rate:6.1%}"
        )


if __name__ == "__main__":
    main()
