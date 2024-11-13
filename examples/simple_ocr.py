import argparse
import glob
import json
import os

import cv2

from yomitoku import OCR
from yomitoku.data.functions import load_image


def main(args):
    ocr = OCR(
        visualize=args.vis,
        device=args.device,
    )

    for image in glob.glob(os.path.join(args.image, "*.jpg")):
        img = load_image(image)
        preds, vis = ocr(img)

        os.makedirs(args.outdir, exist_ok=True)

        if vis is not None:
            filename = os.path.basename(image)
            out_vis = os.path.join(args.outdir, f"{filename}")
            cv2.imwrite(out_vis, vis)

        name, _ = os.path.splitext(os.path.basename(image))
        with open(os.path.join(args.outdir, f"{name}_result.json"), "w") as f:
            json.dump(
                preds,
                f,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
                separators=(",", ": "),
            )

    ocr.detector.save_config(os.path.join(args.outdir, "text_detector.yaml"))
    ocr.recognizer.save_config(
        os.path.join(args.outdir, "text_recognizer.yaml")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--det_config", type=str, default=None)
    parser.add_argument("--rec_config", type=str, default=None)
    parser.add_argument("--image", type=str, default="dataset/test_20241013")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    main(args)
