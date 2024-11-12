import argparse
import glob
import os

import cv2

from yomitoku import LayoutParser
from yomitoku.data.functions import load_image


def main(args):
    layout_parser = LayoutParser(visualize=args.vis, device=args.device)

    for image in glob.glob(os.path.join(args.image, "*.jpg")):
        img = load_image(image)
        preds, vis = layout_parser(img)

        os.makedirs(args.outdir, exist_ok=True)

        if vis is not None:
            out_vis = os.path.join(args.outdir, f"{os.path.basename(image)}")
            cv2.imwrite(out_vis, vis)
            print(f"Saved visualization result to {out_vis}")

        layout_parser.save_config(
            os.path.join(args.outdir, "layout_parser.yaml")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout_config", type=str, default=None)
    parser.add_argument("--image", type=str, default="dataset/test_20241013")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--outdir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    main(args)
