import argparse
import os
import cv2
import json
from yomitoku import OCR


def main(args):
    filename = os.path.basename(args.image)
    name, ext = os.path.splitext(filename)

    ocr = OCR(args.config, visualize=args.vis)
    preds, vis = ocr(args.image)

    os.makedirs(args.outdir, exist_ok=True)

    if vis is not None:
        out_vis = os.path.join(args.outdir, f"{name}_visualize.jpg")
        cv2.imwrite(out_vis, vis)

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
    parser.add_argument("--config", type=str, default="configs/ocr.yaml")
    parser.add_argument("--image", type=str, default="dataset/00003126_5721251_5.jpg")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--outdir", type=str, default="results")
    args = parser.parse_args()

    main(args)
