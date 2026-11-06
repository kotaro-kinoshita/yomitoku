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
        # every page found in the image (both pages of a spread, receipts side by side, ...)
        outputs = analyzer.analyze_pages(img)

        dewarper = analyzer.preprocessor.dewarper
        grid_vis_img = dewarper.visualize(img)  # pages found, drawn on the input image
        cv2.imwrite(f"output_dewarp_grid_{i}.jpg", grid_vis_img)

        for k, (results, ocr_vis, layout_vis) in enumerate(outputs):
            dewarp = results.preprocess.dewarp
            print(dewarp.status, dewarp.reasons, dewarp.page, dewarp.num_pages)

            # the image the results refer to (the dewarped page with dewarp_coordinate="dewarped")
            cv2.imwrite(f"output_page_{i}_{k}.jpg", analyzer.processed_imgs[k])
            results.to_json(f"output_{i}_{k}.json")
            cv2.imwrite(f"output_layout_{i}_{k}.jpg", layout_vis)
