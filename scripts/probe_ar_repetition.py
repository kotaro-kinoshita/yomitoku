"""Probe per-step AR continuous values to test decode-time repetition detection.

Mirrors PARSeq.forward's AR loop but records, at every step i:
  conf   = max softmax prob (confidence of the emitted token)
  ent    = entropy of the distribution (nats)
  eos_p  = probability assigned to EOS
  tok    = emitted token id / char
  selfsim= max cosine similarity of the prob vector to a prior step (lag<=8)

Then it checks candidate detectors:
  D1 token-id cycle: a unit of period p (1..8) repeated k times consecutively
  D2 distribution periodicity: selfsim high & stable over a window
  D3 sustained low confidence / EOS suppression

Run on a normal crop (high score) vs repetition-inducing synthetic inputs.
"""

import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from yomitoku.constants import ROOT_DIR
from yomitoku.text_recognizer import TextRecognizer
from yomitoku.data.dataset import ParseqDataset

FONT = ROOT_DIR + "/resource/ShipporiMinchoB1-Bold.ttf"


def canvas(h=32, w=800, g=255):
    return np.full((h, w, 3), g, dtype=np.uint8)


def put(img, text, x, size=24):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text((x, 4), text, font=ImageFont.truetype(FONT, size), fill=(0, 0, 0))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


@torch.inference_mode()
def ar_probe(model, img_tensor, max_steps=100):
    """Re-implements the AR loop, capturing per-step stats. No early stop."""
    dev = next(model.parameters()).device
    images = img_tensor.to(dev)
    bs = images.shape[0]
    num_steps = max_steps + 1
    memory = model.encode(images)
    pos_queries = model.pos_queries[:, :num_steps].expand(bs, -1, -1)
    tgt_mask = torch.triu(
        torch.ones((num_steps, num_steps), dtype=torch.bool, device=dev), 1
    )
    tgt_in = torch.full((bs, num_steps), model.tokenizer.pad_id, dtype=torch.long, device=dev)
    tgt_in[:, 0] = model.tokenizer.bos_id

    eos_id = model.tokenizer.eos_id
    probs_hist = []  # softmax vectors
    rec = []  # per-step dict
    for i in range(num_steps):
        j = i + 1
        tgt_out = model.decode(
            tgt_in[:, :j], memory, tgt_mask[:j, :j],
            tgt_query=pos_queries[:, i:j], tgt_query_mask=tgt_mask[i:j, :j],
        )
        logits = model.head(tgt_out)[:, -1]  # (bs, C) last position
        prob = F.softmax(logits, -1)[0]  # bs=1
        conf, tok = prob.max(-1)
        ent = -(prob * (prob + 1e-12).log()).sum()
        eos_p = prob[eos_id]
        # self-similarity to prior steps (lag 1..8)
        selfsim, simlag = 0.0, 0
        for lag in range(1, min(8, len(probs_hist)) + 1):
            c = F.cosine_similarity(prob, probs_hist[-lag], dim=0).item()
            if c > selfsim:
                selfsim, simlag = c, lag
        probs_hist.append(prob)
        tok_id = int(tok)
        rec.append(dict(
            i=i, tok=tok_id, conf=float(conf), ent=float(ent),
            eos_p=float(eos_p), selfsim=selfsim, simlag=simlag,
        ))
        if j < num_steps:
            tgt_in[:, j] = tok_id
        if tok_id == eos_id:
            break
    return rec


def char_of(model, tok):
    try:
        return model.tokenizer._itos[tok]
    except Exception:
        return "?"


def detect_cycle(toks, p_max=8, k_min_p1=8, k_min_multi=3):
    """First step index where a period-p unit has repeated k times (decode-time)."""
    n = len(toks)
    for end in range(1, n + 1):
        seq = toks[:end]
        for p in range(1, p_max + 1):
            if len(seq) < 2 * p:
                continue
            unit = seq[-p:]
            k = 1
            t = len(seq) - p
            while t - p >= 0 and seq[t - p:t] == unit:
                k += 1
                t -= p
            kmin = k_min_p1 if p == 1 else k_min_multi
            if k >= kmin:
                return end - 1, p, k  # step index at detection
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="parseq-large-v4_1")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    rec = TextRecognizer(model_name=args.model_name, device=args.device)
    model = rec.model
    cfg = rec._cfg

    cases = [
        ("normal", put(canvas(32, 300), "請求書", 6)),
        ("blank", canvas(32, 800)),
        ("rule-line", cv2.line(canvas(32, 800), (10, 16), (790, 16), (0, 0, 0), 2)),
        ("edge-text", put(canvas(32, 800), "東京", 2)),
    ]

    def to_tensor(img):
        ds = ParseqDataset(cfg, img, [[[0, 0], [img.shape[1], 0],
                                       [img.shape[1], img.shape[0]], [0, img.shape[0]]]])
        return ds[0].unsqueeze(0)

    for name, img in cases:
        steps = ar_probe(model, to_tensor(img), max_steps=cfg.max_label_length)
        toks = [s["tok"] for s in steps]
        text = "".join(char_of(model, t) for t in toks if t != model.tokenizer.eos_id)
        cyc = detect_cycle(toks)
        print(f"\n===== {name}  (len={len(steps)})  -> {text[:50]!r} =====")
        det = f"step {cyc[0]} period{cyc[1]}x{cyc[2]}" if cyc else "none"
        print(f"  D1 token-cycle detect: {det}")
        # show a window around onset
        lo = 0 if not cyc else max(0, cyc[0] - 4)
        hi = min(len(steps), lo + 16)
        print(f"  {'i':>3} {'char':>5} {'conf':>5} {'ent':>5} {'eosP':>5} {'selfsim':>7}")
        for s in steps[lo:hi]:
            ch = char_of(model, s["tok"]).replace("\n", "\\n")
            print(f"  {s['i']:>3} {ch:>5} {s['conf']:.3f} {s['ent']:.2f} "
                  f"{s['eos_p']:.3f}  {s['selfsim']:.3f}@{s['simlag']}")
        arr = np.array([(s["conf"], s["ent"], s["selfsim"]) for s in steps])
        print(f"  mean conf={arr[:,0].mean():.3f}  mean ent={arr[:,1].mean():.2f}  "
              f"mean selfsim={arr[:,2].mean():.3f}")


if __name__ == "__main__":
    main()
