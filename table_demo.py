from yomitoku.table_semantic_parser import TableSemanticParser

from yomitoku.data.functions import load_image

import cv2
import os


analyzer = TableSemanticParser(
    device="cuda:1",
    visualize=True,
)


# path_img = "MIND-form.png"
# path_img = "dataset/table_parser/マーカス九品寺_１０８号室_0.jpg"
path_img = "dataset/table_parser/00075397_7321904_0_0.jpg"

# path_img = "dataset/table_parser/CAA_document_public_1_0a__10.jpg"

# path_img = "dataset/table_parser/Screenshot_from_2025-03-24_22-44-18_0.jpg"
# path_img = "dataset/202407120400_0004.png"
#
# path_img = "dataset/table_parser/IY_225甲子園店_20250314_鮮魚_1_0_0.jpg"
# path_img = "dataset/train_20241016_diversity/0001/00002336_3394946_0.jpg"
# path_img = "dataset/test_20241014_good/0000/00016723_3293630_3.jpg"
# path_img = "dataset/table_parser/00008357_3484180_4.jpg"

# path_img = "dataset/table_parser/202407120135_0000_0.jpg"
# path_img = "dataset/table_parser/test2_7.jpg"
# path_img = "dataset/table_parser/test2_7.jpg"
# path_img = "dataset/table_parser/個人申告書サンプル_1.jpg"

# path_img = "dataset/test_20241014_good/0000/00035702_3323442_0.jpg"
imgs = load_image(path_img)

# path_img = "3-1.適格請求書発行事業者の登録申請書（MLism株式会社御中）.PDF"
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

    cv2.imwrite(f"{output}/{basename}_{i}_ocr.jpg", vis_ocr)
    cv2.imwrite(f"{output}/{basename}_{i}_cell.jpg", vis_cell)
    cv2.imwrite(f"{output}/{basename}_{i}_group.jpg", vis_group)

    # results.to_json(f"{output}/table_analyzer_result_{i}.json")

    # print(results.search_cell_by_query(query="金額"))
    # print(results.search_cell_by_id(table_id=1, cell_id=1).contents)
    results.to_json(f"{output}/{basename}_raw_result_{i}.json")

    # d = results.to_dict()
    # with open(f"{output}/{basename}_result_{i}.json", "w") as f:
    #    json.dump(d, f, indent=4, ensure_ascii=False)

    # print(results.search_table_by_id(table_id=1))

    # print(results.search_cell_by_key("項目"))
    # print(results.search_kv_items_by_key_query("電話番号"))
    # print(results.search_kv_items_by_key_query("氏名"))
    # print(results.search_kv_items_by_key_query("住所"))

    # print(
    table = results.find_table_by_id(table_id=0)

    # print(table.find_cell_by_id(cell_id=1))
    # print(table.search_cells_by_query(query="代表者氏名"))
    # print(table.search_cells_by_query(query="法人番号"))

    # print(table.search_cells_below_key_text(key="アクセスレベル"))
    # print(table.search_cells_below_key_text(key=""))
    # print(table.search_cells_right_of_key_text(key="賃貸条件"))
    # print(table.search_kv_items_by_key(key="建物"))
    # print(table.search_cells_by_bbox([355, 322, 418, 353]))
    # rint(table.find_kv_items_by_key(query="申請日"))
    # print(table.find_kv_items_by_key(query="氏名"))
    # print(table.find_kv_items_by_key(query="所属"))

    # print(export.grids_to_json(f"{output}/{basename}_grid_{i}.json"))

    table.export.grids_to_csv(
        # columns=["鋼管", "合計"],
        out_path=f"{output}/{basename}_table_{i}.csv",
    )

    # table.export.grids_to_csv(f"{output}/{basename}_grids_{i}.csv", grid_id=1)

    # print(table.export.grids_to_dataframe().head())
    # print(export.kv_items_to_dataframe())

    # table.export.kv_items_to_json(f"{output}/{basename}_kv_items_{i}.json")
    # able.export.kv_items_to_csv(f"{output}/{basename}_kv_items_{i}.csv")
    # print(table.view.kv_items_to_dict())
    print(table.view.grids_to_dicts())

    # print(
    #    export.kv_items_to_csv(
    #        out_path=f"{output}/{basename}_kv_{i}.csv",
    #    )
    # )

    # results.kv_items_to_csv(out_path="table_kv_dataframe.csv")

    ##print(pd.DataFrame(parsed["matrix"][0]))

    # import pprint
    # pprint.pprint(parsed)
