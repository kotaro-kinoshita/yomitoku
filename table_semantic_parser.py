from yomitoku.table_semantic_parser import TableSemanticParser

from yomitoku.data.functions import load_image

import cv2
import os
import json

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
        results, vis_cell, vis_group, vis_ocr = analyzer(img)

        if len(results.tables) == 0:
            print(f"No table found in {path_img} page {i}")
            continue

        cv2.imwrite(f"{output}/{basename}_cell.jpg", vis_cell)
        cv2.imwrite(f"{output}/{basename}_group.jpg", vis_group)
        cv2.imwrite(f"{output}/{basename}_ocr.jpg", vis_ocr)

        # results.to_json(f"{output}/table_analyzer_result_{i}.json")

        d = results.to_dict()

        with open(f"{output}/{basename}_result_{i}.json", "w") as f:
            json.dump(d, f, indent=4, ensure_ascii=False)

        # dataframe = results.to_dataframe()
        #
        # dataframe["pairs"].to_csv(f"{output}/{basename}_pairs_{i}.csv", index=False)
        # for j, df in enumerate(dataframe["matrix"]):
        #    print(df.head())
        #    df.to_csv(f"{output}/{basename}_{j}.csv", index=False)

        ##print(pd.DataFrame(parsed["matrix"][0]))

        # import pprint
        # pprint.pprint(parsed)
