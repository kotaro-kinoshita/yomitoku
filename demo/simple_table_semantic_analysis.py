import cv2
from pprint import pprint

from yomitoku.table_semantic_parser import TableSemanticParser
from yomitoku.data.functions import load_image

path_img = "demo/samples/table.jpg"

analyzer = TableSemanticParser(
    device="cuda",
    visualize=True,
)

imgs = load_image(path_img)

results, vis_layout, vis_ocr = analyzer(
    imgs[0],
)

# 解析結果をJSONで保存
results.to_json("result.json")

# OCRの読み取り結果を可視化
cv2.imwrite("vis_ocr.jpg", vis_ocr)

# セル検出結果を可視化
cv2.imwrite("vis_cell.jpg", vis_layout)


# 利用情報を表示
# t0というIDのテーブルを取得
table = results.find_table_by_id(table_id="t0")

# テーブルのすべてのスキーマを表示
pprint(table.view.kv_items_to_dict())
"""
>>{'利用目的': 'セミナー', '実施内容': 'YomiTokuの利用方法に関する説明会', '施設名称': 'MLism株式会社'}
"""

pprint(table.view.grids_to_dict())
"""
>>[{'id': 'g0',
>>  'rows': [{'入室時刻': '10時00分', '日付': '2025年01月30日(月曜日)', '退室時刻': '17時00分'},
>>           {'入室時刻': '10時00分', '日付': '2025年02月1日(火曜日)', '退室時刻': '17時00分'},
>>           {'入室時刻': '時分', '日付': '年月日(曜日)', '退室時刻': '時分'},
>>           {'入室時刻': '時分', '日付': '年月日(曜日)', '退室時刻': '時分'},
>>           {'入室時刻': '時分', '日付': '年月日(曜日)', '退室時刻': '時分'}]
>>}]
"""


# 解析結果から団体名を検索して表示
table = results.find_table_by_id(table_id="t1")
pprint(table.search_kv_items_by_key(key="団体名"))
"""
>> [{'key': [CellSchema(meta={}, contents='団 体 名', role='header', id='c1', box=[149, 1393, 364, 1539], row=None, col=None, row_span=None, col_span=None)],
>>  'value': CellSchema(meta={}, contents='エムリズムカブシキガイシャ', role='cell', id='c2', box=[365, 1393, 1498, 1444], row=None, col=None, row_span=None, col_span=None)},
>> {'key': [CellSchema(meta={}, contents='団 体 名', role='header', id='c1', box=[149, 1393, 364, 1539], row=None, col=None, row_span=None, col_span=None)],
>>  'value': CellSchema(meta={}, contents='MLism 株式会社', role='cell', id='c3', box=[365, 1444, 1498, 1539], row=None, col=None, row_span=None, col_span=None)}]
"""


# 解析結果から住所を検索して表示
pprint(table.search_kv_items_by_key(key="住所"))
"""
[{'key': [CellSchema(meta={}, contents='住 所', role='header', id='c7', box=[149, 1686, 364, 1788], row=None, col=None, row_span=None, col_span=None)],
  'value': CellSchema(meta={}, contents='〒277-8520千葉県柏市若柴 178 番地 4 柏の葉キャンパス 148 街区 2 ショップ&オフィス棟 6F', role='cell', id='c8', box=[365, 1686, 1498, 1788], row=None, col=None, row_span=None, col_span=None)}]
"""
