import cv2
import os

from yomitoku.table_semantic_parser import TableSemanticParser

from yomitoku.data.functions import load_pdf


analyzer = TableSemanticParser(
    device="cuda:1",
    visualize=True,
)


path_img = "dataset/EPSON012.PDF"
imgs = load_pdf(path_img)
output = "outputs"

os.makedirs(output, exist_ok=True)

basename = os.path.splitext(os.path.basename(path_img))[0]

for i, img in enumerate(imgs):
    results, vis_cell, vis_ocr = analyzer(
        img,
    )

    if len(results.tables) == 0:
        print(f"No table found in {path_img} page {i}")
        continue

    # OCRの読み取り結果を可視化
    cv2.imwrite(f"{output}/{basename}_{i}_ocr.jpg", vis_ocr)

    # セル検出結果を可視化
    cv2.imwrite(f"{output}/{basename}_{i}_cell.jpg", vis_cell)

    # 解析結果をJSONで保存
    results.to_json(f"{output}/{basename}_raw_result_{i}.json")

    # 解析結果から住所というキーを持つKey-Valueペアを検索
    address_kv = results.search_kv_items_by_key(key="住所")
    print(address_kv)
    """
    [{'key': [CellSchema(meta={}, contents='住 所', role='header', id='c10', box=[155, 702, 371, 861])], 'value': [CellSchema(meta={}, contents='〒260 277-8520千葉県柏市若菜178番地4柏の葉キャンパス148街区2ショップ&オフィス棟6下', role='cell', id='c11', box=[371, 682, 1508, 858])]}]
    """

    # テーブルIDがt0のテーブルを取得
    table = results.find_table_by_id(table_id="t0")

    # テーブルt0から「申込日」というキーを持つKey-Valueペアを検索
    print(table.search_kv_items_by_key(key="申込日"))
    """
    [{'key': [CellSchema(meta={}, contents='申込日', role='header', id='c0', box=[838, 464, 1053, 529])], 'value': [CellSchema(meta={}, contents='2026年1月18日', role='cell', id='c1', box=[1053, 456, 1505, 525])]}]
    """

    # テーブルt0に含まれるKey-Valueペアを辞書形式で取得
    print(table.view.kv_items_to_dict())
    """
    {'申込日': '2026年1月18日', 'フリガナ_0': 'サトウ タロウ', 'フリガナ_1': 'エムリズム', '申込者名': '佐藤太郎', '団体名': 'MLism株式会社', '住 所': '〒260 277-8520千葉県柏市若菜178番地4柏の葉キャンパス148街区2ショップ&オフィス棟6下', '連絡先_0': 'TEL : 090-1234-5678 FAX :', '連絡先_1': '携帯: E-mail: test@mlism-com'}
    """

    # テーブルt0 に含まれるKey-ValueをJSONで保存
    table.export.kv_items_to_json(f"{output}/{basename}_kv_items_{i}.json")

    # テーブルt0 に含まれるKey-ValueをcSVで保存
    table.export.kv_items_to_csv(f"{output}/{basename}_kv_items_{i}.csv")

    # テーブルIDがt1のテーブルを取得
    table = results.find_table_by_id(table_id="t1")

    # テーブルt1から「イベント名」というキーにその右隣のセルを検索
    print(table.search_cells_right_of_key_text(key="イベント名"))
    """
     [CellSchema(meta={}, contents='Document AI開発', role='cell', id='c1', box=[380, 1045, 1513, 1165])]
    """

    # テーブルt1から「イベント名」というキーにその右隣のセルを検索
    print(table.search_cells_right_of_key_text(key="内容"))
    """
    [CellSchema(meta={}, contents='頑張ってAIを開発する会', role='cell', id='c3', box=[381, 1163, 1513, 1261])]
    """

    # テーブルt1に対して、指定したバウンディングボックス内にあるセルを検索
    print(table.search_cells_by_bbox([380, 1045, 1513, 1165]))
    """
    [CellSchema(meta={}, contents='Document AI開発', role='cell', id='c1', box=[380, 1045, 1513, 1165])]
    """

    table = results.find_table_by_id(table_id="t2")

    # テーブルt2に含まれるグリッド情報をJSONで保存
    table.export.grids_to_json(f"{output}/{basename}_grids_{i}.json")

    # テーブルt2に含まれるグリッド情報をDataFrameで取得して先頭5行を表示
    print(table.export.grids_to_dataframe().head())
    """
    使 用 日     開始時間    終了時間 備考
    0  2026年1月20日  8 時 00分  23時00分   
    1  2026年1月21日   8 時00分  23時00分   
    2       年 月 日      時 分     時 分   
    3       年 月 日      時 分     時 分   
    4       年 月 日      時 分     時 分 
    """

    # テーブルt2に含まれるグリッド情報をCSVで保存。指定したカラムのみ抽出。
    table.export.grids_to_csv(
        columns=["使用日", "開始時間", "終了時間"],
        out_path=f"{output}/{basename}_table_{i}.csv",
    )

    """
    ,使 用 日,開始時間,終了時間
    0,2026年1月20日,8 時 00分,23時00分
    1,2026年1月21日,8 時00分,23時00分
    2,年 月 日,時 分,時 分
    3,年 月 日,時 分,時 分
    4,年 月 日,時 分,時 分
    ,,,
    """

    # テンプレートJSONとして保存
    results.save_template_json(f"{output}/{basename}_template_{i}.json")

    # テンプレートJSONを使って再解析
    results, vis_cell, vis_ocr = analyzer(
        img,
        template=f"{output}/{basename}_template_{i}.json",
    )
