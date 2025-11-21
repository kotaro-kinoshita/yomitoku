import cv2

from yomitoku import OCR
from yomitoku.data.functions import load_pdf

if __name__ == "__main__":
    ocr = OCR(visualize=True, device="cuda")
    imgs = load_pdf("demo/sample.pdf")

    for i, img in enumerate(imgs):
        results, ocr_vis = ocr(img)

        results.to_json(f"output_{i}.json")
        cv2.imwrite(f"output_ocr_{i}.jpg", ocr_vis)
