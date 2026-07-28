"""table_semantic_parser の検証スクリプト (ローカル学習モデル / 全体表示版).

rtdetrv2_pytorch で学習した最新モデル (configs の weights_path で指定) を使って
データセット画像を「クロップせず画像全体」に対して解析し、
- セル + 解析済み grid の可視化 (*_semantic.jpg)
- モデルが直接予測した kv_item / grid 領域の可視化 (*_regions.jpg)
- OCR 可視化 (*_ocr.jpg)
- 構造化結果 (*_result.json)
を出力する。

Usage:
    python demo/validate_table_semantic_parser.py \
        -i /home/user/Projects/rtdetrv2_pytorch/dataset/table_semantic_parser/images/dataset/merged_table_parser/images/train \
        -o output/validate_semantic \
        -n 20 -d cuda

    # 入力画像サイズ 960 の検証用モデルを使う場合
    python demo/validate_table_semantic_parser.py -m rtdetrv2_beta_960 ...
"""

import argparse
import glob
import os

import cv2

from yomitoku.data import load_pdf
from yomitoku.table_semantic_parser import (
    TableSemanticParser,
    _resolve_overlapping_regions,
)

DEFAULT_INPUT = (
    "/home/user/Projects/rtdetrv2_pytorch/dataset/table_semantic_parser/"
    "images/dataset/merged_table_parser/images/train"
)

# kv_item / grid 領域の描画色 (BGR)
REGION_COLORS = {
    "kv_item": (0, 140, 255),  # orange
    "grid": (200, 0, 200),  # magenta
}


def draw_raw_regions(analyzer, img):
    """モデルが直接予測した kv_item / grid 領域を画像全体に重ねて描画する。

    重複解決の前(細線)と後(太線)を色分けして、どちらが採用されたか確認できる。
    """
    vis = img.copy()

    # レイアウト解析でテーブル領域を取得し、各テーブルに対してセル検出を実行
    results_layout, _ = analyzer.layout_parser(img)
    tables = list(results_layout.tables)
    detected = analyzer.cell_detector(img, tables)

    for det in detected:
        value_cells = [
            c for c in det.cells if c.role in ("cell", "header", "empty")
        ]
        grid_regions = list(det.grid_regions)
        kv_regions = list(det.kv_regions)

        kept_grids, kept_kvs = _resolve_overlapping_regions(
            grid_regions, kv_regions, value_cells
        )
        kept_grid_boxes = {tuple(g.box) for g in kept_grids}
        kept_kv_boxes = {tuple(k.box) for k in kept_kvs}

        for region in grid_regions:
            kept = tuple(region.box) in kept_grid_boxes
            _draw_region(vis, region, REGION_COLORS["grid"], kept)

        for region in kv_regions:
            kept = tuple(region.box) in kept_kv_boxes
            _draw_region(vis, region, REGION_COLORS["kv_item"], kept)

    return vis


def _draw_region(vis, region, color, kept):
    x1, y1, x2, y2 = region.box
    thickness = 3 if kept else 1
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
    tag = f"{region.role}:{region.score:.2f}{'' if kept else ' (dropped)'}"
    cv2.putText(
        vis,
        tag,
        (x1 + 2, max(0, y1 - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


def main(args):
    analyzer = TableSemanticParser(
        configs={
            "table_cell_parser": {"model_name": args.cell_detector_model},
            "table_detector": {"model_name": args.layout_model},
        },
        device=args.device,
        visualize=True,
    )

    if os.path.isfile(args.input):
        files = [args.input]
    else:
        # フォルダ指定時はサブディレクトリも再帰的に探索する
        files = []
        for ext in ("*.jpg", "*.png", "*.pdf"):
            files += glob.glob(os.path.join(args.input, "**", ext), recursive=True)
        files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No images/PDFs found in {args.input}")

    if args.num > 0 and len(files) > args.num:
        step = max(1, len(files) // args.num)
        files = files[::step][: args.num]

    os.makedirs(args.output, exist_ok=True)
    print(f"processing {len(files)} files -> {args.output}\n")

    for f in files:
        # 再帰探索でサブフォルダ間のファイル名衝突を避けるため、
        # 入力ディレクトリからの相対パスを "__" で連結した名前にする
        if os.path.isdir(args.input):
            rel = os.path.relpath(f, args.input)
            base = os.path.splitext(rel)[0].replace(os.sep, "__")
        else:
            base = os.path.splitext(os.path.basename(f))[0]

        if f.lower().endswith(".pdf"):
            # PDF はページごとに画像化して解析する
            try:
                pages = load_pdf(f, dpi=args.dpi)
            except Exception as e:
                print(f"[skip] cannot load {f}: {e}")
                continue
            for i, img in enumerate(pages):
                _process_one(analyzer, args, f"{base}_p{i}", img)
        else:
            img = cv2.imread(f)
            if img is None:
                print(f"[skip] cannot read {f}")
                continue
            _process_one(analyzer, args, base, img)

    print(f"\nDONE -> {args.output}")


def _process_one(analyzer, args, name, img):
    results, vis_layout, vis_ocr = analyzer(img, grid_only=False)

    # 画像全体の可視化を保存
    cv2.imwrite(os.path.join(args.output, f"{name}_semantic.jpg"), vis_layout)
    cv2.imwrite(os.path.join(args.output, f"{name}_ocr.jpg"), vis_ocr)
    results.to_json(os.path.join(args.output, f"{name}_result.json"))

    if args.regions:
        vis_regions = draw_raw_regions(analyzer, img)
        cv2.imwrite(os.path.join(args.output, f"{name}_regions.jpg"), vis_regions)

    n_tables = len(results.tables)
    n_grids = sum(len(t.grids) for t in results.tables)
    n_kv = sum(len(t.kv_items) for t in results.tables)
    print(
        f"{name}: tables={n_tables} grids={n_grids} kv_items={n_kv} "
        f"(shape={img.shape[0]}x{img.shape[1]})"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "-i",
        "--input",
        default=DEFAULT_INPUT,
        help="画像/PDF ファイル、またはそれらを含むディレクトリ(再帰探索)",
    )
    p.add_argument("-o", "--output", default="output/validate_semantic")
    p.add_argument(
        "-n", "--num", type=int, default=20, help="処理する画像数 (0で全部)"
    )
    p.add_argument("-d", "--device", default="cuda")
    p.add_argument(
        "-m",
        "--cell-detector-model",
        default="rtdetrv2_beta",
        choices=["rtdetrv2_beta", "rtdetrv2_beta_960"],
        help=(
            "セル検出モデルのバリアント "
            "(rtdetrv2_beta: img_size 640, rtdetrv2_beta_960: img_size 960 検証用)"
        ),
    )
    p.add_argument(
        "--layout-model",
        default="rtdetrv2v2",
        choices=["rtdetrv2", "rtdetrv2v2"],
        help="テーブル検出(layout_parser)モデルのバリアント",
    )
    p.add_argument(
        "--dpi", type=int, default=200, help="PDF をページ画像化する際の解像度"
    )
    p.add_argument(
        "--regions",
        action="store_true",
        help="モデルが直接予測した kv_item / grid 領域も可視化する",
    )
    args = p.parse_args()
    main(args)
