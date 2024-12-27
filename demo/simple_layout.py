import cv2

from yomitoku import LayoutAnalyzer
from yomitoku.data.functions import load_pdf

if __name__ == "__main__":
    analyzer = LayoutAnalyzer(visualize=True, device="cuda")
    # PDFファイルを読み込み
    imgs = load_pdf("demo/sample.pdf")
    for i, img in enumerate(imgs):
        results, layout_vis = analyzer(img)

        # JSON形式で解析結果をエクスポート
        results.to_json(f"output_{i}.json")
        cv2.imwrite(f"output_layout_{i}.jpg", layout_vis)
