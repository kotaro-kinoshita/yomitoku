"""Feed synthetic crops through PARSeq and analyze the repetition patterns.

Goal: empirically observe *what* the recognizer emits for the failure-mode
inputs reported in practice, instead of guessing the output patterns.

Synthetic scenarios (rendered as 32xW BGR crops, points=None -> whole image):
  S1  text with large left/right blank margin
  S1b text pushed to one edge, rest blank
  S2  pure blank canvas
  S2b blank + random smudges / speckle noise
  S2c logo-like geometric shapes (no text)
  S3  random texture / untrained non-text
  S3b thin horizontal rule / underline only
  ctrl tight normal text crop (control)

For each output we print: content (repr), score, length, whether it hit
max_label_length (EOS failure), and the smallest repeating unit + run length.
"""

import argparse

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from yomitoku.constants import ROOT_DIR
from yomitoku.text_recognizer import TextRecognizer

FONT_PATH = ROOT_DIR + "/resource/ShipporiMinchoB1-Bold.ttf"


def _canvas(h=32, w=800, gray=255):
    return np.full((h, w, 3), gray, dtype=np.uint8)


def _put_text(img, text, x, y=4, size=24, color=(0, 0, 0)):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = ImageFont.truetype(FONT_PATH, size)
    draw.text((x, y), text, font=font, fill=color[::-1])
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def make_cases(seed=0):
    rng = np.random.default_rng(seed)
    cases = []

    # S1: small text, large horizontal margin (text ~ left third, rest blank)
    img = _put_text(_canvas(32, 800), "売上高", x=20)
    cases.append(("S1 margin-wide", img))

    # S1b: text pushed to far left edge
    img = _put_text(_canvas(32, 800), "東京", x=2)
    cases.append(("S1b edge-text", img))

    # S2: pure blank
    cases.append(("S2 blank", _canvas(32, 800)))

    # S2b: blank + speckle noise / smudges
    img = _canvas(32, 800)
    noise = rng.integers(0, 80, size=(32, 800), dtype=np.int16)
    mask = rng.random((32, 800)) < 0.04
    img[mask] = np.clip(255 - noise[mask, None], 0, 255)
    # a couple of gray smudge blobs
    for _ in range(3):
        cx, cy = int(rng.integers(50, 750)), int(rng.integers(8, 24))
        cv2.circle(img, (cx, cy), int(rng.integers(4, 9)), (120, 120, 120), -1)
    cases.append(("S2b smudge", img))

    # S2c: logo-like geometric shapes, no text
    img = _canvas(32, 800)
    cv2.rectangle(img, (30, 6), (70, 26), (0, 0, 0), 2)
    cv2.circle(img, (120, 16), 11, (0, 0, 0), 2)
    cv2.line(img, (160, 6), (200, 26), (0, 0, 0), 3)
    cv2.line(img, (200, 6), (160, 26), (0, 0, 0), 3)
    cases.append(("S2c logo", img))

    # S3: random texture (untrained non-text)
    img = rng.integers(0, 255, size=(32, 800, 3), dtype=np.uint8)
    cases.append(("S3 noise-texture", img))

    # S3b: thin horizontal rule only
    img = _canvas(32, 800)
    cv2.line(img, (10, 16), (790, 16), (0, 0, 0), 2)
    cases.append(("S3b rule-line", img))

    # control: tight normal text
    img = _put_text(_canvas(32, 300), "請求書", x=6)
    cases.append(("ctrl normal", img))

    return cases


def smallest_period(s, p_max=20):
    """Return (period, repeats, run_chars) of the longest repeated run (any pos)."""
    n = len(s)
    best = (0, 0, 0)  # run_chars, period, repeats
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="parseq-large-v4_1")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dynamic_width", action="store_true")
    ap.add_argument("--save_dir", default=None)
    args = ap.parse_args()

    rec = TextRecognizer(
        model_name=args.model_name,
        device=args.device,
        dynamic_width=args.dynamic_width,
    )
    max_len = rec._cfg.max_label_length
    print(f"model={args.model_name} dynamic_width={args.dynamic_width} "
          f"max_label_length={max_len}\n")

    cases = make_cases()
    header = f"{'case':<18}{'len':>4} {'maxlen?':>7} {'score':>7}  {'period':>6}x{'k':<3} content"
    print(header)
    print("-" * 100)
    for name, img in cases:
        if args.save_dir:
            import os
            os.makedirs(args.save_dir, exist_ok=True)
            cv2.imwrite(f"{args.save_dir}/{name.replace(' ', '_')}.png", img)
        results, _ = rec(img, points=None)
        content = results.contents[0]
        score = results.scores[0]
        p, k, run = smallest_period(content)
        hit = "YES" if len(content) >= max_len else ""
        disp = content if len(content) <= 40 else content[:40] + "…"
        print(f"{name:<18}{len(content):>4} {hit:>7} {score:>7.3f}  "
              f"{p:>6}x{k:<3} {disp!r}")


if __name__ == "__main__":
    main()
