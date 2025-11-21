import cv2

from yomitoku import DocumentAnalyzer
from yomitoku.data.functions import load_pdf

if __name__ == "__main__":
    PATH_IMGE = "demo/sample.pdf"
    analyzer = DocumentAnalyzer(visualize=True, device="cuda")
    imgs = load_pdf(PATH_IMGE)

    for i, img in enumerate(imgs):
        results, ocr_vis, layout_vis = analyzer(img)

        results.to_html(f"output_{i}.html", img=img)

        cv2.imwrite(f"output_ocr_{i}.jpg", ocr_vis)
        cv2.imwrite(f"output_layout_{i}.jpg", layout_vis)
