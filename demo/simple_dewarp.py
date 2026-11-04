import cv2

from yomitoku import DocumentAnalyzer
from yomitoku.data.functions import load_image

if __name__ == "__main__":
    PATH_IMGE = "static/in/gallery4.jpg"
    analyzer = DocumentAnalyzer(
        visualize=True,
        device="cuda",
        enable_dewarp=True,
        dewarp_coordinate="dewarped",  # "photo": map the results back to the input image
    )
    imgs = load_image(PATH_IMGE)

    for i, img in enumerate(imgs):
        results, ocr_vis, layout_vis = analyzer(img)

        dewarp = results.preprocess.dewarp
        print(dewarp.status, dewarp.reasons)

        dewarper = analyzer.preprocessor.dewarper
        if dewarp.status == "ok":
            dewarped_img = dewarper.last_dewarped_img
            cv2.imwrite(f"output_dewarped_{i}.jpg", dewarped_img)

        grid_vis_img = dewarper.visualize(
            img
        )  # paper region and grid on the input image
        cv2.imwrite(f"output_dewarp_grid_{i}.jpg", grid_vis_img)

        results.to_json(f"output_{i}.json")
        cv2.imwrite(f"output_layout_{i}.jpg", layout_vis)
