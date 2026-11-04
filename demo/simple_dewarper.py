import cv2

from yomitoku import Dewarper
from yomitoku.data.functions import load_image

if __name__ == "__main__":
    PATH_IMGE = "static/in/gallery4.jpg"
    dewarper = Dewarper(device="cuda")
    imgs = load_image(PATH_IMGE)

    for i, img in enumerate(imgs):
        results, dewarped_img = dewarper(img)
        print(results.status, results.reasons, results.size)

        if results.status == "ok":
            cv2.imwrite(f"output_dewarped_{i}.jpg", dewarped_img)

        grid_vis_img = dewarper.visualize(
            img
        )  # paper region and grid on the input image
        cv2.imwrite(f"output_dewarp_grid_{i}.jpg", grid_vis_img)
