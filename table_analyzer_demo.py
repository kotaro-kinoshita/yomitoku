from yomitoku.table_analyzer import TableAnalyzer

from yomitoku.data.functions import load_image

import cv2
import os


analyzer = TableAnalyzer(
    device="cuda",
    visualize=True,
)

path_img = "Image-001_0.jpg"
#
# path_img = "00075397_7321904_0_0.jpg"
# path_img = "00008357_3484180_4.jpg"
# path_img = "dataset/202407120135_0000.png"
# path_img = "CAA_document_public_1_0a__10.jpg"
imgs = load_image(path_img)

# path_img = "朝日プラザ鍛冶屋町_３０７号室.pdf"
# imgs = load_pdf(path_img)
output = "outputs"


os.makedirs(output, exist_ok=True)

for i, img in enumerate(imgs):
    cv2.imwrite(f"{os.path.splitext(os.path.basename(path_img))[0]}_{i}.jpg", img)
    results, vis_cell, vis_group = analyzer(img, i)
    cv2.imwrite(f"{output}/table_analyzer_cell{i}.jpg", vis_cell)
    cv2.imwrite(f"{output}/table_analyzer_group{i}.jpg", vis_group)
