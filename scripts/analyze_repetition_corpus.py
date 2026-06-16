"""Build a ~1000+ crop corpus from real images and analyze repetition trends.

Sources per image (from static/in):
  real    : crops at the TextDetector quads (legitimate-text baseline)
  margin  : same quads widened left/right (induces the large-margin failure)
  nontext : random boxes (blank / figure / logo / untrained non-text)

For every crop we run the recognizer and record content, score, length, the
smallest repeated unit (period/repeats) and whether it hit max_label_length.
Then we aggregate: score distributions, repetition rates by source and by
score band, max-length-hit rate, and separability of legit vs failure crops.
"""

import argparse
import csv
import glob

import numpy as np
import torch

from yomitoku.text_detector import TextDetector
from yomitoku.text_recognizer import TextRecognizer
from yomitoku.data.functions import load_image


def quad_to_bbox(q):
    q = np.asarray(q)
    return q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()


def bbox_to_quad(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def widen(q, w, factor, rng):
    x0, y0, x1, y1 = quad_to_bbox(q)
    bw = x1 - x0
    m = bw * rng.uniform(factor * 0.5, factor)
    nx0 = max(0, int(x0 - m))
    nx1 = min(w - 1, int(x1 + m))
    return bbox_to_quad(nx0, y0, nx1, y1)


def random_box(w, h, widths, heights, rng):
    bw = int(rng.choice(widths))
    bh = int(rng.choice(heights))
    x0 = int(rng.integers(0, max(1, w - bw)))
    y0 = int(rng.integers(0, max(1, h - bh)))
    return bbox_to_quad(x0, y0, min(w - 1, x0 + bw), min(h - 1, y0 + bh))


def smallest_period(s, p_max=20):
    """(period, repeats, run_chars) of the longest repeated run anywhere."""
    n = len(s)
    best = (0, 0, 0)
    for i in range(n):
        for p in range(1, p_max + 1):
            if i + 2 * p > n:
                break
            unit = s[i : i + p]
            k = 1
            while s[i + k * p : i + (k + 1) * p] == unit:
                k += 1
            if k >= 2 and p * k > best[0]:
                best = (p * k, p, k)
    return best[1], best[2], best[0]


def recognize(rec, img, quads):
    if not quads:
        return [], []
    results, _ = rec(img, quads)
    return list(results.contents), list(results.scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="parseq-large-v4_1")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--target", type=int, default=1000)
    ap.add_argument("--nontext", type=int, default=400)
    ap.add_argument("--out_csv", default="/tmp/repetition_corpus.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    det = TextDetector(device=args.device)
    rec = TextRecognizer(model_name=args.model_name, device=args.device)
    max_len = rec._cfg.max_label_length

    rows = []  # dict per sample

    def add(source, img_name, content, score):
        p, k, run = smallest_period(content)
        rows.append(
            dict(
                source=source,
                image=img_name,
                content=content,
                score=float(score),
                length=len(content),
                period=p,
                repeats=k,
                run=run,
                hit_max=int(len(content) >= max_len),
            )
        )

    images = sorted(glob.glob("static/in/*.jp*g"))
    # first pass: detect + collect width/height stats for nontext boxes
    cache = []
    all_w, all_h = [], []
    for path in images:
        name = path.split("/")[-1]
        img = load_image(path)[0]
        res, _ = det(img)
        quads = res.points
        cache.append((name, img, quads))
        for q in quads:
            x0, y0, x1, y1 = quad_to_bbox(q)
            all_w.append(max(4, x1 - x0))
            all_h.append(max(4, y1 - y0))
    all_w = np.array(all_w)
    all_h = np.array(all_h)

    # real
    for name, img, quads in cache:
        c, s = recognize(rec, img, quads)
        for ci, si in zip(c, s):
            add("real", name, ci, si)

    # margin (widen each quad)
    for name, img, quads in cache:
        h, w = img.shape[:2]
        wq = [widen(q, w, factor=rng.uniform(1.0, 4.0), rng=rng) for q in quads]
        c, s = recognize(rec, img, wq)
        for ci, si in zip(c, s):
            add("margin", name, ci, si)

    # nontext random boxes (fixed quota) + top up to target
    n_nontext = 0
    target_rows = len(rows) + args.nontext
    while len(rows) < target_rows or len(rows) < args.target:
        for name, img, quads in cache:
            h, w = img.shape[:2]
            boxes = [random_box(w, h, all_w, all_h, rng) for _ in range(20)]
            c, s = recognize(rec, img, boxes)
            for ci, si in zip(c, s):
                add("nontext", name, ci, si)
                n_nontext += 1
            if len(rows) >= target_rows and len(rows) >= args.target:
                break

    # ---- write csv ----
    with open(args.out_csv, "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)

    # ---- aggregate ----
    import collections

    print(f"\nmodel={args.model_name}  samples={len(rows)}  max_label_length={max_len}")
    print(f"csv -> {args.out_csv}\n")

    def rep_kind(r):
        if r["run"] >= 6 and r["period"] == 1:
            return "char-loop"
        if r["repeats"] >= 3 and r["period"] >= 2:
            return "word-loop"
        return "none"

    for r in rows:
        r["rep"] = rep_kind(r)

    print("=== by source ===")
    print(f"{'source':<9}{'n':>5}{'score<.3':>9}{'char-loop':>10}"
          f"{'word-loop':>10}{'hit_max':>8}{'med_score':>10}")
    for src in ["real", "margin", "nontext"]:
        sub = [r for r in rows if r["source"] == src]
        if not sub:
            continue
        n = len(sub)
        lo = sum(r["score"] < 0.3 for r in sub)
        cl = sum(r["rep"] == "char-loop" for r in sub)
        wl = sum(r["rep"] == "word-loop" for r in sub)
        hm = sum(r["hit_max"] for r in sub)
        med = float(np.median([r["score"] for r in sub]))
        print(f"{src:<9}{n:>5}{lo:>9}{cl:>10}{wl:>10}{hm:>8}{med:>10.3f}")

    print("\n=== repetition vs score band (all samples) ===")
    bands = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 1.01)]
    print(f"{'band':<12}{'n':>6}{'char-loop':>10}{'word-loop':>10}{'rep%':>7}")
    for lo, hi in bands:
        sub = [r for r in rows if lo <= r["score"] < hi]
        if not sub:
            continue
        cl = sum(r["rep"] == "char-loop" for r in sub)
        wl = sum(r["rep"] == "word-loop" for r in sub)
        pct = 100 * (cl + wl) / len(sub)
        print(f"[{lo:.2f},{hi:.2f}){'':<2}{len(sub):>6}{cl:>10}{wl:>10}{pct:>6.0f}%")

    print("\n=== repetition kind counts ===")
    print(collections.Counter(r["rep"] for r in rows))

    # threshold separability: legit(real) vs failure(margin+nontext)
    print("\n=== score threshold: legit(real) kept vs failure(margin+nontext) dropped ===")
    real_s = np.array([r["score"] for r in rows if r["source"] == "real"])
    fail_s = np.array([r["score"] for r in rows if r["source"] != "real"])
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        kept = (real_s >= t).mean() * 100
        dropped = (fail_s < t).mean() * 100
        print(f"  t={t:.1f}: real kept {kept:5.1f}%   failure dropped {dropped:5.1f}%")

    print("\n=== sample repetition outputs (score<0.3) ===")
    shown = 0
    for r in rows:
        if r["rep"] != "none" and r["score"] < 0.3:
            disp = r["content"][:48] + ("…" if len(r["content"]) > 48 else "")
            print(f"  [{r['source']:<7} p{r['period']}x{r['repeats']:<3} "
                  f"s{r['score']:.2f}] {disp!r}")
            shown += 1
            if shown >= 25:
                break


if __name__ == "__main__":
    main()
