"""Render analysis figures from the repetition corpus CSV."""

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["score"] = float(r["score"])
            r["length"] = int(r["length"])
            r["period"] = int(r["period"])
            r["repeats"] = int(r["repeats"])
            r["run"] = int(r["run"])
            rows.append(r)
    return rows


def rep_kind(r):
    if r["run"] >= 6 and r["period"] == 1:
        return "char-loop"
    if r["repeats"] >= 3 and r["period"] >= 2:
        return "word-loop"
    return "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/tmp/repetition_corpus.csv")
    ap.add_argument("--out", default="/tmp/repetition_analysis.png")
    args = ap.parse_args()

    rows = load(args.csv)
    for r in rows:
        r["rep"] = rep_kind(r)
    sources = ["real", "margin", "nontext"]
    colors = {"real": "#2e7d32", "margin": "#ef6c00", "nontext": "#c62828"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"PARSeq repetition analysis  (n={len(rows)}, model=parseq-large-v4_1)",
        fontsize=14, fontweight="bold",
    )

    # (1) score histogram by source
    ax = axes[0, 0]
    bins = np.linspace(0, 1, 26)
    for src in sources:
        s = [r["score"] for r in rows if r["source"] == src]
        ax.hist(s, bins=bins, alpha=0.6, label=f"{src} (n={len(s)})", color=colors[src])
    ax.set_title("(1) Score distribution by source")
    ax.set_xlabel("recognition score")
    ax.set_ylabel("count")
    ax.legend()
    ax.set_yscale("log")

    # (2) repetition rate vs score band
    ax = axes[0, 1]
    bands = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 1.01)]
    labels, char_pct, word_pct = [], [], []
    for lo, hi in bands:
        sub = [r for r in rows if lo <= r["score"] < hi]
        n = max(1, len(sub))
        labels.append(f"[{lo:.1f},{hi:.1f})\nn={len(sub)}")
        char_pct.append(100 * sum(r["rep"] == "char-loop" for r in sub) / n)
        word_pct.append(100 * sum(r["rep"] == "word-loop" for r in sub) / n)
    x = np.arange(len(labels))
    ax.bar(x, char_pct, 0.5, label="char-loop", color="#1565c0")
    ax.bar(x, word_pct, 0.5, bottom=char_pct, label="word-loop", color="#6a1b9a")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("(2) Repetition rate vs score band")
    ax.set_ylabel("% of band that is a loop")
    ax.legend()

    # (3) scatter score vs length, colored by rep kind
    ax = axes[1, 0]
    kind_color = {"none": "#bdbdbd", "char-loop": "#1565c0", "word-loop": "#6a1b9a"}
    rng = np.random.default_rng(0)
    for kind in ["none", "char-loop", "word-loop"]:
        sub = [r for r in rows if r["rep"] == kind]
        xs = [r["score"] + rng.uniform(-0.004, 0.004) for r in sub]
        ys = [r["length"] for r in sub]
        ax.scatter(xs, ys, s=10, alpha=0.5, c=kind_color[kind],
                   label=f"{kind} (n={len(sub)})")
    ax.axvline(0.1, ls="--", c="k", lw=1)
    ax.text(0.105, ax.get_ylim()[1] * 0.9, "score=0.1", fontsize=8)
    ax.set_title("(3) Score vs output length")
    ax.set_xlabel("recognition score")
    ax.set_ylabel("output length (chars)")
    ax.legend()

    # (4) threshold tradeoff curve
    ax = axes[1, 1]
    real_s = np.array([r["score"] for r in rows if r["source"] == "real"])
    fail_s = np.array([r["score"] for r in rows if r["source"] != "real"])
    ts = np.linspace(0, 0.6, 61)
    kept = [(real_s >= t).mean() * 100 for t in ts]
    dropped = [(fail_s < t).mean() * 100 for t in ts]
    ax.plot(ts, kept, label="real (legit) kept %", color="#2e7d32", lw=2)
    ax.plot(ts, dropped, label="failure dropped %", color="#c62828", lw=2)
    for t in [0.1, 0.2, 0.3]:
        ax.axvline(t, ls=":", c="gray", lw=1)
    ax.set_title("(4) Score threshold tradeoff")
    ax.set_xlabel("threshold τ")
    ax.set_ylabel("%")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=130)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
