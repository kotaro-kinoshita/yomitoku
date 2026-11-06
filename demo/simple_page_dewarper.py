import cv2

from yomitoku import PageDewarper
from yomitoku.data.functions import load_image

if __name__ == "__main__":
    PATH_IMGE = "static/in/gallery4.jpg"
    dewarper = PageDewarper(device="cuda")
    imgs = load_image(PATH_IMGE)

    for i, img in enumerate(imgs):
        # one (results, image) per page found, left to right then top to bottom
        for results, dewarped_img in dewarper(img):
            print(results.status, results.page, results.num_pages, results.score)
            if results.status == "ok":
                cv2.imwrite(f"output_dewarped_{i}_{results.page}.jpg", dewarped_img)

        grid_vis_img = dewarper.visualize(img)  # pages found, drawn on the input image
        cv2.imwrite(f"output_dewarp_grid_{i}.jpg", grid_vis_img)
