from yomitoku.table_semantic_parser import TableSemanticParser

from yomitoku.data.functions import load_image

import cv2
import os

analyzer = TableSemanticParser(
    device="cuda:1",
    visualize=True,
)

tardir = "dataset/test_20241014_good/0000"

for file in os.listdir(tardir):
    # path_img = "dataset/table_parser/Image-001_0_0.jpg"
    #
    # path_img = "dataset/table_parser/個人申告書サンプル_0.jpg"
    # path_img = "dataset/train_20241016_diversity/0001/00002336_3394946_0.jpg"
    # path_img = "dataset/test_20241014_good/0000/00061385_2714926_166.jpg"
    # path_img = "dataset/table_parser/00008357_3484180_4.jpg"
    # path_img = "dataset/table_parser/マーカス九品寺_１０８号室_0.jpg"
    # path_img = "dataset/table_parser/202407120135_0000_0.jpg"
    # path_img = "dataset/table_parser/CAA_document_public_1_0a__10.jpg"
    # path_img = "dataset/table_parser/test2_7.jpg"

    path_img = os.path.join(tardir, file)
    imgs = load_image(path_img)

    # path_img = "dataset/様式9号16-1.pdf"
    # path_img = "dataset/20250625_sample/docs/sample.pdf"
    # imgs = load_pdf(path_img)
    output = "outputs"

    os.makedirs(output, exist_ok=True)

    basename = os.path.splitext(os.path.basename(path_img))[0]

    for i, img in enumerate(imgs):
        # cv2.imwrite(f"{os.path.splitext(os.path.basename(path_img))[0]}_{i}.jpg", img)
        results, vis_cell, vis_ocr = analyzer(
            img,
            # template=f"{output}/{basename}_raw_result_{i}.json",
            # f"{output}/{basename}_template_{i}.json",
        )

        if len(results.tables) == 0:
            print(f"No table found in {path_img} page {i}")
            continue

        results.save_template_json(f"{output}/{basename}_template_{i}.json")

        cv2.imwrite(f"{output}/{basename}_{i}_ocr.jpg", vis_ocr)
        cv2.imwrite(f"{output}/{basename}_{i}_cell.jpg", vis_cell)

        vis_items = img.copy()
        vis_grids = img.copy()

        for j, table in enumerate(results.tables):
            vis_grids = table.view.visualize_grids(vis_grids)
            vis_items = table.view.visualize_kv_items(vis_items)

        cv2.imwrite(
            f"{output}/{basename}_vis_grids_{i}.jpg",
            vis_grids,
        )
        cv2.imwrite(
            f"{output}/{basename}_kv_items_{i}.jpg",
            vis_items,
        )

        table.export.kv_items_to_json(f"{output}/{basename}_kv_items_{i}.json")
        table.export.grids_to_json(f"{output}/{basename}_grids_{i}.json")
        table.export.grids_to_csv(f"{output}/{basename}_grids_{i}.csv")
        table.export.kv_items_to_csv(f"{output}/{basename}_kv_items_{i}.csv")
