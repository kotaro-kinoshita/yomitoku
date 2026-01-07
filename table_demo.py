from yomitoku.table_semantic_parser import TableSemanticParser

from yomitoku.data.functions import load_image

import cv2
import os


analyzer = TableSemanticParser(
    device="cuda:1",
    visualize=True,
)


# path_img = "dataset/table_parser/00075397_7321904_0_0_0.jpg"
# path_img = "dataset/table_parser/マーカス九品寺_１０８号室_0.jpg"
#
# path_img = "dataset/table_parser/IY_225甲子園店_20250314_鮮魚_1_0_0.jpg"
path_img = "dataset/train_20241016_diversity/0001/00002336_3394946_0.jpg"
# path_img = "dataset/test_20241014_good/0000/00016723_3293630_3.jpg"
# path_img = "dataset/table_parser/00008357_3484180_4.jpg"

# path_img = "dataset/table_parser/202407120135_0000_0.jpg"
# path_img = "dataset/table_parser/test2_7.jpg"
# path_img = "dataset/table_parser/test2_7.jpg"
# path_img = "dataset/table_parser/個人申告書サンプル_1.jpg"

# path_img = "dataset/test_20241014_good/0000/00035702_3323442_0.jpg"
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

    cv2.imwrite(f"{output}/{basename}_{i}.jpg", img)
    cv2.imwrite(f"{output}/{basename}_{i}_cell.jpg", vis_cell)
    cv2.imwrite(f"{output}/{basename}_{i}_group.jpg", vis_group)

    # results.to_json(f"{output}/table_analyzer_result_{i}.json")

    # print(results.search_cell_by_query(query="金額"))
    # print(results.search_cell_by_id(table_id=1, cell_id=1).contents)
    print

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
    table = results.search_table_by_id(table_id=0)

    # print(table.search_cell_by_id(cell_id=1))
    print(table.search_cell_by_query(query="住所"))
    print(table.search_kv_items_by_key(query="製品名"))

    # print(export.grids_to_json(f"{output}/{basename}_grid_{i}.json"))

    # table.export.grids_to_csv(
    #    columns=["鋼管", "合計"],
    #    out_path=f"{output}/{basename}_table_{i}.csv",
    # )

    # print(table.export.grids_to_dataframe().head())
    # print(export.kv_items_to_dataframe())

    table.export.kv_items_to_json(f"{output}/{basename}_kv_items_{i}.json")
    table.export.kv_items_to_csv(f"{output}/{basename}_kv_items_{i}.csv")
    print(table.view.kv_items_to_dict())

    # print(
    #    export.kv_items_to_csv(
    #        out_path=f"{output}/{basename}_kv_{i}.csv",
    #    )
    # )

    # dataframe = results.to_dataframe()
    # results.grid_to_csv(
    #    table_id=1,
    #    columns=["対象梱包", "区分", "間口方向"],
    #    out_path="table_dataframe.csv",
    # )
    # results.kv_items_to_csv(out_path="table_kv_dataframe.csv")

    ##print(pd.DataFrame(parsed["matrix"][0]))

    # import pprint
    # pprint.pprint(parsed)
