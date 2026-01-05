from yomitoku.table_semantic_parser import TableSemanticParser

from yomitoku.data.functions import load_image

import cv2
import os
import json

analyzer = TableSemanticParser(
    device="cuda:1",
    visualize=True,
)

# path_img = "dataset/table_parser/Image-001_0_0.jpg"
#
# path_img = "dataset/table_parser/個人申告書サンプル_7.jpg"
path_img = "dataset/test_20241014_good/0000/00016723_3293630_3.jpg"
# path_img = "dataset/test_20241014_good/0000/00061385_2714926_166.jpg"
# path_img = "dataset/table_parser/00008357_3484180_4.jpg"
# path_img = "dataset/table_parser/マーカス九品寺_１０８号室_0.jpg"
# path_img = "dataset/table_parser/202407120135_0000_0.jpg"
# path_img = "dataset/table_parser/CAA_document_public_1_0a__10.jpg"
# path_img = "dataset/table_parser/00075397_7321904_0_0.jpg"
imgs = load_image(path_img)

# path_img = "dataset/security_q2403.pdf"
# path_img = "dataset/20250625_sample/docs/sample.pdf"
# imgs = load_pdf(path_img)
output = "outputs"


os.makedirs(output, exist_ok=True)

for i, img in enumerate(imgs):
    # cv2.imwrite(f"{os.path.splitext(os.path.basename(path_img))[0]}_{i}.jpg", img)
    results, vis_cell, vis_group, vis_ocr = analyzer(img, i)
    cv2.imwrite(f"{output}/table_analyzer_cell{i}.jpg", vis_cell)
    cv2.imwrite(f"{output}/table_analyzer_group{i}.jpg", vis_group)

    with open(f"{output}/key_values_{i}.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
