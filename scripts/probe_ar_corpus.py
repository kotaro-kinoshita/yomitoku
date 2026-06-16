"""Corpus-scale evaluation of decode-time repetition detectors.

For each crop we run an instrumented AR loop (NO early stop) and record per
step: emitted token id, confidence (max softmax prob), and the best
distribution self-similarity cos(softmax_t, softmax_{t-lag}) over lag 1..8.

Then we evaluate, offline (so thresholds can be swept), three decode-time
detectors and compare them to the post-hoc ground-truth "is a loop":
  A token-cycle      : a period-p unit repeats k times (p1 strict, multi loose)
  B dist-periodicity : selfsim >= theta sustained for W consecutive steps
  C confidence       : per-step conf < c for W consecutive steps (expected to fail)

Reports: precision/recall vs GT, mean AR-step savings, and the false-positive
rate on legitimate 'real' crops (the safety-critical metric).
"""

import argparse
import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from yomitoku.text_detector import TextDetector
from yomitoku.text_recognizer import TextRecognizer
from yomitoku.data.dataset import ParseqDataset
from yomitoku.data.functions import load_image


# ----- ground-truth repetition label (post-hoc, on final string) -----
def smallest_period(s, p_max=20):
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
    return best[1], best[2], best[0]  # period, repeats, run


def is_loop(text):
    p, k, run = smallest_period(text)
    return (p == 1 and run >= 6) or (p >= 2 and k >= 3)


# ----- instrumented AR loop -----
@torch.inference_mode()
def ar_trace(model, img_tensor, max_steps):
    dev = next(model.parameters()).device
    images = img_tensor.to(dev)
    num_steps = max_steps + 1
    memory = model.encode(images)
    pos_q = model.pos_queries[:, :num_steps].expand(1, -1, -1)
    m = torch.triu(torch.ones((num_steps, num_steps), dtype=torch.bool, device=dev), 1)
    tgt = torch.full((1, num_steps), model.tokenizer.pad_id, dtype=torch.long, device=dev)
    tgt[:, 0] = model.tokenizer.bos_id
    eos = model.tokenizer.eos_id
    hist, toks, confs, sims = [], [], [], []
    for i in range(num_steps):
        j = i + 1
        out = model.decode(tgt[:, :j], memory, m[:j, :j],
                           tgt_query=pos_q[:, i:j], tgt_query_mask=m[i:j, :j])
        prob = F.softmax(model.head(out)[:, -1], -1)[0]
        conf, tk = prob.max(-1)
        best = 0.0
        for lag in range(1, min(8, len(hist)) + 1):
            c = F.cosine_similarity(prob, hist[-lag], dim=0).item()
            if c > best:
                best = c
        hist.append(prob)
        tk = int(tk)
        toks.append(tk)
        confs.append(float(conf))
        sims.append(best)
        if j < num_steps:
            tgt[:, j] = tk
        if tk == eos:
            break
    return toks, confs, sims


# ----- decode-time detectors (return fire step index or None) -----
def det_token_cycle(toks, p_max=8, k1=8, km=3):
    for e in range(1, len(toks) + 1):
        seq = toks[:e]
        for p in range(1, p_max + 1):
            if len(seq) < 2 * p:
                continue
            unit = seq[-p:]
            k, t = 1, len(seq) - p
            while t - p >= 0 and seq[t - p : t] == unit:
                k += 1
                t -= p
            if k >= (k1 if p == 1 else km):
                return e - 1
    return None


def det_sustained(values, thresh, window, ge=True):
    run = 0
    for idx, v in enumerate(values):
        ok = v >= thresh if ge else v < thresh
        run = run + 1 if ok else 0
        if run >= window:
            return idx
    return None


def char_of(model, t):
    try:
        return model.tokenizer._itos[t]
    except Exception:
        return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="parseq-large-v4_1")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n_real", type=int, default=250)
    ap.add_argument("--n_margin", type=int, default=250)
    ap.add_argument("--n_nontext", type=int, default=120)
    ap.add_argument("--images_glob", default="static/in/*.jp*g")
    ap.add_argument("--max_images", type=int, default=0, help="0 = all")
    ap.add_argument("--fig", default=None, help="output figure path")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    det = TextDetector(device=args.device)
    rec = TextRecognizer(model_name=args.model_name, device=args.device)
    model, cfg = rec.model, rec._cfg
    eos = model.tokenizer.eos_id
    max_len = cfg.max_label_length

    # collect quads from real images
    cache = []
    ws, hs = [], []
    paths = sorted(glob.glob(args.images_glob))
    if args.max_images and len(paths) > args.max_images:
        idx = np.linspace(0, len(paths) - 1, args.max_images).astype(int)
        paths = [paths[i] for i in idx]
    print(f"images: {len(paths)} (from {args.images_glob})")
    for path in paths:
        try:
            img = load_image(path)[0]
            res, _ = det(img)
        except Exception:
            continue
        if len(res.points) == 0:
            continue
        cache.append((img, res.points))
        for q in res.points:
            q = np.asarray(q)
            ws.append(max(4, q[:, 0].max() - q[:, 0].min()))
            hs.append(max(4, q[:, 1].max() - q[:, 1].min()))
    ws, hs = np.array(ws), np.array(hs)

    def crop_tensor(img, quad):
        ds = ParseqDataset(cfg, img, [quad])
        if len(ds) == 0:
            return None
        return ds[0].unsqueeze(0)

    def widen(q, w):
        q = np.asarray(q)
        x0, y0, x1, y1 = q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()
        mrg = (x1 - x0) * rng.uniform(1.0, 4.0)
        nx0, nx1 = max(0, int(x0 - mrg)), min(w - 1, int(x1 + mrg))
        return [[nx0, y0], [nx1, y0], [nx1, y1], [nx0, y1]]

    def rand_box(w, h):
        bw, bh = int(rng.choice(ws)), int(rng.choice(hs))
        x0, y0 = int(rng.integers(0, max(1, w - bw))), int(rng.integers(0, max(1, h - bh)))
        return [[x0, y0], [x0 + bw, y0], [x0 + bw, y0 + bh], [x0, y0 + bh]]

    samples = []  # (source, quad_provider)
    all_quads = [(img, q) for img, qs in cache for q in qs]
    rng.shuffle(all_quads)
    for img, q in all_quads[: args.n_real]:
        samples.append(("real", img, q))
    for img, q in all_quads[: args.n_margin]:
        samples.append(("margin", img, widen(q, img.shape[1])))
    nt = 0
    while nt < args.n_nontext:
        img = cache[int(rng.integers(len(cache)))][0]
        samples.append(("nontext", img, rand_box(img.shape[1], img.shape[0])))
        nt += 1

    rows = []
    for n, (src, img, quad) in enumerate(samples):
        ten = crop_tensor(img, quad)
        if ten is None:
            continue
        toks, confs, sims = ar_trace(model, ten, max_len)
        body = [t for t in toks if t != eos]
        text = "".join(char_of(model, t) for t in body)
        rows.append(dict(src=src, text=text, toks=toks, confs=confs, sims=sims,
                         steps=len(toks), gt=is_loop(text)))
        if (n + 1) % 100 == 0:
            print(f"  ...{n + 1}/{len(samples)} traced")

    print(f"\nmodel={args.model_name}  crops={len(rows)}  max_len={max_len}")
    by = {s: [r for r in rows if r["src"] == s] for s in ["real", "margin", "nontext"]}
    for s, sub in by.items():
        gt = sum(r["gt"] for r in sub)
        print(f"  {s:<8} n={len(sub):<4} GT-loops={gt}")

    # selfsim separability
    def feat(r):
        sims = r["sims"]
        return max(sims) if sims else 0.0, det_sustained(sims, 0.9, 1) is not None

    loop = [r for r in rows if r["gt"]]
    nonloop = [r for r in rows if not r["gt"]]
    print("\n=== selfsim separability (max self-similarity over steps) ===")
    for name, grp in [("loop", loop), ("non-loop", nonloop)]:
        v = np.array([max(r["sims"]) if r["sims"] else 0 for r in grp])
        print(f"  {name:<9} n={len(grp):<4} median={np.median(v):.3f} "
              f"p10={np.percentile(v,10):.3f} p90={np.percentile(v,90):.3f}")

    # evaluate detectors
    def evaluate(name, fire_fn):
        tp = fp = fn = 0
        savings, fp_real = [], 0
        for r in rows:
            fs = fire_fn(r)
            fired = fs is not None
            if r["gt"] and fired:
                tp += 1
                savings.append(r["steps"] - 1 - fs)
            elif r["gt"] and not fired:
                fn += 1
            elif (not r["gt"]) and fired:
                fp += 1
                if r["src"] == "real":
                    fp_real += 1
        prec = tp / (tp + fp) if tp + fp else 0
        recl = tp / (tp + fn) if tp + fn else 0
        ms = np.mean(savings) if savings else 0
        print(f"  {name:<28} P={prec:.2f} R={recl:.2f}  "
              f"FP={fp}(real {fp_real})  mean_step_saved={ms:.0f}")

    print("\n=== decode-time detector comparison ===")
    evaluate("A token-cycle (p1>=8,m>=3)",
             lambda r: det_token_cycle(r["toks"], k1=8, km=3))
    evaluate("A token-cycle (p1>=10,m>=3)",
             lambda r: det_token_cycle(r["toks"], k1=10, km=3))
    for th, w in [(0.97, 3), (0.95, 3), (0.95, 4), (0.90, 4)]:
        evaluate(f"B dist-sim (>={th}, W={w})",
                 lambda r, th=th, w=w: det_sustained(r["sims"], th, w))
    for c, w in [(0.3, 6), (0.5, 8)]:
        evaluate(f"C conf (<{c}, W={w})",
                 lambda r, c=c, w=w: det_sustained(r["confs"], c, w, ge=False))

    # FP examples on real
    print("\n=== false positives on 'real' crops (dist-sim>=0.95,W=4) ===")
    shown = 0
    for r in rows:
        if r["src"] == "real" and not r["gt"] and det_sustained(r["sims"], 0.95, 4) is not None:
            print(f"  {r['text'][:50]!r}")
            shown += 1
            if shown >= 10:
                break
    if shown == 0:
        print("  (none)")

    if args.fig:
        make_figure(rows, args.fig, args.model_name,
                    lambda r: det_token_cycle(r["toks"], k1=8, km=3))
        print(f"\nsaved figure -> {args.fig}")


def make_figure(rows, out, model_name, detA):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loop = [r for r in rows if r["gt"]]
    nonloop = [r for r in rows if not r["gt"]]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Decode-time repetition detection on board_vqa "
                 f"(n={len(rows)}, model={model_name})", fontsize=13, fontweight="bold")

    # (1) max selfsim distribution
    ax = axes[0, 0]
    bins = np.linspace(0, 1, 26)
    ax.hist([max(r["sims"]) if r["sims"] else 0 for r in nonloop], bins=bins,
            alpha=0.6, color="#bdbdbd", label=f"non-loop (n={len(nonloop)})")
    ax.hist([max(r["sims"]) if r["sims"] else 0 for r in loop], bins=bins,
            alpha=0.7, color="#c62828", label=f"loop (n={len(loop)})")
    ax.set_title("(1) max distribution self-similarity")
    ax.set_xlabel("max cos(softmax_t, softmax_{t-lag})")
    ax.set_ylabel("count"); ax.set_yscale("log"); ax.legend()

    # (2) per-step selfsim traces (sample)
    ax = axes[0, 1]
    for r in loop[:25]:
        ax.plot(r["sims"], color="#c62828", alpha=0.35, lw=0.8)
    for r in nonloop[:25]:
        ax.plot(r["sims"], color="#9e9e9e", alpha=0.35, lw=0.8)
    ax.plot([], [], color="#c62828", label="loop")
    ax.plot([], [], color="#9e9e9e", label="non-loop")
    ax.set_title("(2) per-step selfsim traces (<=25 each)")
    ax.set_xlabel("AR step"); ax.set_ylabel("selfsim"); ax.legend()

    # (3) detector comparison
    ax = axes[1, 0]
    def evald(fire):
        tp = fp = fn = 0
        for r in rows:
            f = fire(r) is not None
            tp += r["gt"] and f
            fn += r["gt"] and not f
            fp += (not r["gt"]) and f
        P = tp / (tp + fp) if tp + fp else 0
        R = tp / (tp + fn) if tp + fn else 0
        return P, R, fp
    dets = [
        ("A token-cycle", detA),
        ("B sim>=.95,W4", lambda r: det_sustained(r["sims"], 0.95, 4)),
        ("C conf<.5,W8", lambda r: det_sustained(r["confs"], 0.5, 8, ge=False)),
    ]
    names = [d[0] for d in dets]
    Ps, Rs, FPs = zip(*[evald(d[1]) for d in dets])
    x = np.arange(len(names))
    ax.bar(x - 0.2, Ps, 0.4, label="Precision", color="#1565c0")
    ax.bar(x + 0.2, Rs, 0.4, label="Recall", color="#2e7d32")
    for i, fpv in enumerate(FPs):
        ax.text(i, 1.02, f"FP={fpv}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 1.15); ax.set_title("(3) detector precision/recall")
    ax.legend(loc="lower right")

    # (4) step savings histogram for detector A
    ax = axes[1, 1]
    sav = []
    for r in rows:
        if r["gt"]:
            fs = detA(r)
            if fs is not None:
                sav.append(r["steps"] - 1 - fs)
    if sav:
        ax.hist(sav, bins=20, color="#ef6c00")
        ax.axvline(np.mean(sav), ls="--", c="k",
                   label=f"mean={np.mean(sav):.0f}")
        ax.legend()
    ax.set_title("(4) AR steps saved by early-stop (detector A)")
    ax.set_xlabel("steps saved"); ax.set_ylabel("loops")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
