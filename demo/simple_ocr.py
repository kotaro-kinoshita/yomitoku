import cv2

from yomitoku import OCR
from yomitoku.data.functions import load_pdf

if __name__ == "__main__":
    ocr = OCR(visualize=True, device="cuda")
    # PDFファイルを読み込み
    imgs = load_pdf("demo/sample.pdf")
    results, ocr_vis = ocr(imgs)

    for i, (result, vis) in enumerate(zip(results, ocr_vis)):
        # JSON形式で解析結果をエクスポート
        result.to_json(f"output_{i}.json")
        cv2.imwrite(f"output_ocr_{i}.jpg", vis)

    # JSON形式で解析結果をエクスポート
    # results.to_json(f"output_{i}.json")
    # cv2.imwrite(f"output_ocr_{i}.jpg", ocr_vis)
