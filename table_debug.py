from yomitoku.table_semantic_parser import TableSemanticParser

from yomitoku.data.functions import load_image

import cv2
import os

analyzer = TableSemanticParser(
    device="cuda:1",
    visualize=True,
)

tardir = "dataset/table_parser"

# for file in os.listdir(tardir):
#    path_img = os.path.join(tardir, file)
#    imgs = load_image(path_img)
#    output = "debug"
#    os.makedirs(output, exist_ok=True)
#    basename = os.path.splitext(os.path.basename(path_img))[0]#

#    print(basename)
#    for i, img in enumerate(imgs):
#        # cv2.imwrite(f"{os.path.splitext(os.path.basename(path_img))[0]}_{i}.jpg", img)
#        results, vis_cell, vis_ocr = analyzer(
#            img,
#            # template=f"{output}/{basename}_raw_result_{i}.json",
#            # f"{output}/{basename}_template_{i}.json",
#        )
#        if len(results.tables) == 0:
#            print(f"No table found in {path_img} page {i}")
#            continue
#        cv2.imwrite(f"{output}/{basename}_{i}_region.jpg", vis_cell)

path_img = "dataset/table_parser/test_0_0.jpg"
imgs = load_image(path_img)
output = "debug"
os.makedirs(output, exist_ok=True)
basename = os.path.splitext(os.path.basename(path_img))[0]
for img in imgs:
    results, vis_cell, vis_ocr = analyzer(
        img,
        # template=f"{output}/{basename}_raw_result_{i}.json",
        # f"{output}/{basename}_template_{i}.json",
    )  #
    if len(results.tables) == 0:
        print(f"No table found in {path_img}")
    cv2.imwrite(f"{output}/{basename}_region.jpg", vis_cell)
