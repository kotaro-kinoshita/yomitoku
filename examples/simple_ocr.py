import argparse
import json
import os

import cv2

from yomitoku import OCR
from yomitoku.data.functions import load_image


def main(args):
    filename = os.path.basename(args.image)
    name, ext = os.path.splitext(filename)

    ocr = OCR(
        args.det_config,
        args.rec_config,
        visualize=args.vis,
        device=args.device,
    )

    img = load_image(args.image)
    preds, vis = ocr(img)

    os.makedirs(args.outdir, exist_ok=True)

    if vis is not None:
        out_vis = os.path.join(args.outdir, f"{name}_visualize.jpg")
        cv2.imwrite(out_vis, vis)

    ocr.detector.save_config(os.path.join(args.outdir, "text_detector.yaml"))
    ocr.recognizer.save_config(
        os.path.join(args.outdir, "text_recognizer.yaml")
    )

    with open(os.path.join(args.outdir, f"{name}_result.json"), "w") as f:
        json.dump(
            preds,
            f,
            ensure_ascii=False,
            indent=4,
            sort_keys=True,
            separators=(",", ": "),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--det_config", type=str, default=None)
    parser.add_argument("--rec_config", type=str, default=None)
    parser.add_argument(
        "--image", type=str, default="dataset/00048896_1207136_5.jpg"
    )
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    main(args)
