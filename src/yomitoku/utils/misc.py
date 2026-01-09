import cv2
import math

from typing import List, Tuple, Dict, Any


import numpy as np

Point = Tuple[float, float]
Poly = List[Point]
Rect = Tuple[float, float, float, float]  # x1,y1,x2,y2


EPS = 1e-9


def load_charset(charset_path):
    with open(charset_path, "r", encoding="utf-8") as f:
        charset = f.read()
    return charset


def filter_by_flag(elements, flags):
    assert len(elements) == len(flags)
    return [element for element, flag in zip(elements, flags) if flag]


def save_image(img, path):
    success, buffer = cv2.imencode(".jpg", img)
    if not success:
        raise ValueError("Failed to encode image")

    with open(path, "wb") as f:
        f.write(buffer.tobytes())


def calc_overlap_ratio(rect_a, rect_b):
    intersection = calc_intersection(rect_a, rect_b)
    if intersection is None:
        return 0, None

    ix1, iy1, ix2, iy2 = intersection

    overlap_width = ix2 - ix1
    overlap_height = iy2 - iy1
    bx1, by1, bx2, by2 = rect_b

    b_area = (bx2 - bx1) * (by2 - by1)
    overlap_area = overlap_width * overlap_height

    overlap_ratio = overlap_area / b_area
    return overlap_ratio, intersection


def is_contained(rect_a, rect_b, threshold=0.8):
    """二つの矩形A, Bが与えられたとき、矩形Bが矩形Aに含まれるかどうかを判定する。
    ずれを許容するため、重複率求め、thresholdを超える場合にTrueを返す。


    Args:
        rect_a (np.array): x1, y1, x2, y2
        rect_b (np.array): x1, y1, x2, y2
        threshold (float, optional): 判定の閾値. Defaults to 0.9.

    Returns:
        bool: 矩形Bが矩形Aに含まれる場合True
    """

    overlap_ratio, _ = calc_overlap_ratio(rect_a, rect_b)

    if overlap_ratio > threshold:
        return True

    return False


def calc_intersection(rect_a, rect_b):
    ax1, ay1, ax2, ay2 = map(int, rect_a)
    bx1, by1, bx2, by2 = map(int, rect_b)

    # 交差領域の左上と右下の座標
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    overlap_width = max(0, ix2 - ix1)
    overlap_height = max(0, iy2 - iy1)

    if overlap_width == 0 or overlap_height == 0:
        return None

    return [ix1, iy1, ix2, iy2]


def is_intersected_horizontal(rect_a, rect_b, threshold=0.5):
    _, ay1, _, ay2 = map(int, rect_a)
    _, by1, _, by2 = map(int, rect_b)

    # 交差領域の左上と右下の座標
    iy1 = max(ay1, by1)
    iy2 = min(ay2, by2)

    min_height = min(ay2 - ay1, by2 - by1)

    overlap_height = max(0, iy2 - iy1)

    if (overlap_height / min_height) < threshold:
        return False

    return True


def is_intersected_vertical(rect_a, rect_b):
    ax1, _, ax2, _ = map(int, rect_a)
    bx1, _, bx2, _ = map(int, rect_b)

    # 交差領域の左上と右下の座標
    ix1 = max(ax1, bx1)
    ix2 = min(ax2, bx2)

    overlap_width = max(0, ix2 - ix1)

    if overlap_width == 0:
        return False

    return True


def quad_to_xyxy(quad):
    x1 = min([x for x, _ in quad])
    y1 = min([y for _, y in quad])
    x2 = max([x for x, _ in quad])
    y2 = max([y for _, y in quad])

    return x1, y1, x2, y2


def convert_table_array(table):
    n_rows = table.n_row
    n_cols = table.n_col

    table_array = [["" for _ in range(n_cols)] for _ in range(n_rows)]

    for cell in table.cells:
        row = cell.row - 1
        col = cell.col - 1
        row_span = cell.row_span
        col_span = cell.col_span
        contents = cell.contents

        for i in range(row, row + row_span):
            for j in range(col, col + col_span):
                table_array[i][j] = contents

    return table_array


def convert_table_array_to_dict(table_array, header_row=1):
    n_cols = len(table_array[0])
    n_rows = len(table_array)

    header_cols = []
    for i in range(n_cols):
        header = []
        for j in range(header_row):
            header.append(table_array[j][i])

        if len(header) > 0:
            header_cols.append("_".join(header))
        else:
            header_cols.append(f"col_{i}")

    table_dict = []
    for i in range(header_row, n_rows):
        row_dict = {}
        for j in range(n_cols):
            row_dict[header_cols[j]] = table_array[i][j]
        table_dict.append(row_dict)

    return table_dict


def clamp(t, lo, hi):
    return max(lo, min(hi, t))


def point_to_segment_distance(px, py, ax, ay, bx, by):
    """
    点(px,py)と線分(ax,ay)-(bx,by)の最短距離を計算
    """

    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    denom = abx * abx + aby * aby
    if denom == 0:
        return math.hypot(px - ax, py - ay)
    t = (apx * abx + apy * aby) / denom
    t = clamp(t, 0.0, 1.0)
    cx, cy = ax + t * abx, ay + t * aby
    return math.hypot(px - cx, py - cy)


def right_edge_to_left_edge_dist(A, B):
    """
    Aの右辺とBの左辺の最短距離を計算
    """

    ax1, ay1, ax2, ay2 = A  # A right edge: (ax2,ay1)-(ax2,ay2)
    bx1, by1, bx2, by2 = B  # B left  edge: (bx1,by1)-(bx1,by2)

    # Aの右上 -> B左辺
    d1 = point_to_segment_distance(ax2, ay1, bx1, by1, bx1, by2)

    # Aの右下 -> B左辺
    d2 = point_to_segment_distance(ax2, ay2, bx1, by1, bx1, by2)

    # Bの左上 -> A右辺
    d3 = point_to_segment_distance(bx1, by1, ax2, ay1, ax2, ay2)

    # Bの左下 -> A右辺
    d4 = point_to_segment_distance(bx1, by2, ax2, ay1, ax2, ay2)

    return max(d1, d4), max(d2, d3), max(d3, d4), max(d1, d2)


def top_edge_to_bottom_edge_dist(A, B):
    """
    Aの下辺とBの上辺の最短距離を計算
    """

    ax1, ay1, ax2, ay2 = A  # A bottom edge: (ax1,ay2)-(ax2,ay2)
    bx1, by1, bx2, by2 = B  # B top    edge: (bx1,by1)-(bx2,by1)

    # Aの左下 -> B上辺
    d1 = point_to_segment_distance(ax1, ay2, bx1, by1, bx2, by1)

    # Aの右下 -> B上辺
    d2 = point_to_segment_distance(ax2, ay2, bx1, by1, bx2, by1)

    # Bの左上 -> A下辺
    d3 = point_to_segment_distance(bx1, by1, ax1, ay2, ax2, ay2)

    # Bの右上 -> A下辺
    d4 = point_to_segment_distance(bx2, by1, ax1, ay2, ax2, ay2)

    return max(d1, d4), max(d2, d3), max(d3, d4), max(d1, d2)


def overlap_interval(i1, i2, j1, j2):
    """
    [i1,i2] と [j1,j2] の重なり長
    """
    return max(0.0, min(i2, j2) - max(i1, j1))


def point_distance(p, q):
    px, py = p
    qx, qy = q
    return math.hypot(px - qx, py - qy)


def gap_interval(interval_a, interval_b):
    """
    interval_a = (a1, a2)
    interval_b = (b1, b2)
    2区間の最短距離（重なれば0）
    """
    a1, a2 = interval_a
    b1, b2 = interval_b

    if b2 < a1:
        return a1 - b2
    if a2 < b1:
        return b1 - a2
    return 0.0


def is_right_adjacent(box_a, box_b, threshold=10, overlap_ratio_th=0.5, rule="soft"):
    """
    box_aの右隣にbox_bがあるか判定
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # 1) 方向制約：BがAの右側（最低条件）
    if bx1 < ax1:
        return False

    # 2) 縦方向の重なり制約
    if overlap_interval(ay1, ay2, by1, by2) < overlap_ratio_th * min(
        ay2 - ay1, by2 - by1
    ):
        return False

    # 3) 頂点間の距離の制約
    # Aの右下とBの左上が近すぎる場合は隣接しない
    if point_distance((ax2, ay2), (bx1, by1)) < threshold:
        return False

    # Aの右上とBの左下が近すぎる場合は隣接しない
    if point_distance((ax2, ay1), (bx1, by2)) < threshold:
        return False

    # 4) 頂点と線分距離制約
    d1, d2, d3, d4 = right_edge_to_left_edge_dist(box_a, box_b)
    if rule == "hard":
        if d1 < threshold and d2 < threshold and d3 < threshold and d4 < threshold:
            return True
    elif rule == "soft":
        if d1 < threshold or d2 < threshold or d3 < threshold or d4 < threshold:
            return True

    return False


def is_bottom_adjacent(box_a, box_b, threshold=10, overlap_ratio_th=0.5, rule="soft"):
    """
    box_aの下隣にbox_bがあるか判定
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # 1) 方向制約：BがAの下側（最低条件）
    if by1 < ay1:
        return False

    # 2) 横方向の重なり制約
    if overlap_interval(ax1, ax2, bx1, bx2) < overlap_ratio_th * min(
        ax2 - ax1, bx2 - bx1
    ):
        return False

    # 3) 頂点間の距離の制約
    # Aの右下とBの左上が近すぎる場合は隣接しない
    if point_distance((ax2, ay2), (bx1, by1)) < threshold:
        return False

    # Aの左下とBの右上が近すぎる場合は隣接しない
    if point_distance((ax1, ay2), (bx2, by1)) < threshold:
        return False

    # 4) 頂点間距離制約
    d1, d2, d3, d4 = top_edge_to_bottom_edge_dist(box_a, box_b)

    if rule == "hard":
        if d1 < threshold and d2 < threshold and d3 < threshold and d4 < threshold:
            return True
    elif rule == "soft":
        if d1 < threshold or d2 < threshold or d3 < threshold or d4 < threshold:
            return True

    return False


def signed_area(poly: Poly) -> float:
    if len(poly) < 3:
        return 0.0
    s = 0.0
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + [poly[0]]):
        s += x1 * y2 - x2 * y1
    return 0.5 * s


def poly_area(poly: Poly) -> float:
    return abs(signed_area(poly))


def ensure_ccw(poly: Poly) -> Poly:
    # クリッパはCCW前提に揃える
    return poly if signed_area(poly) >= 0 else list(reversed(poly))


def cross(ax, ay, bx, by) -> float:
    return ax * by - ay * bx


def is_inside_halfplane_ccw(p: Point, a: Point, b: Point) -> bool:
    """
    clip edge a->b が CCW のとき、左側(=内側)を inside とする
    inside: cross(b-a, p-a) >= 0
    """
    px, py = p
    ax, ay = a
    bx, by = b
    return cross(bx - ax, by - ay, px - ax, py - ay) >= -EPS


def line_intersection(p1: Point, p2: Point, a: Point, b: Point) -> Point:
    """
    直線 p1->p2 と 直線 a->b の交点（無限直線として計算）
    Sutherland–Hodgman 用。平行に近い場合は p2 を返して退避。
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = a
    x4, y4 = b

    # (p1 + t*(p2-p1)) と (a + u*(b-a)) の交点
    dx12, dy12 = x2 - x1, y2 - y1
    dx34, dy34 = x4 - x3, y4 - y3

    denom = cross(dx12, dy12, dx34, dy34)
    if abs(denom) < EPS:
        return p2  # ほぼ平行

    # t = cross((a - p1), (b - a)) / cross((p2 - p1), (b - a))
    t = cross(x3 - x1, y3 - y1, dx34, dy34) / denom
    return (x1 + t * dx12, y1 + t * dy12)


def clip_convex_poly(subject: Poly, clipper: Poly) -> Poly:
    """
    凸多角形 subject を 凸多角形 clipper でクリップ（傾きOK）
    ※ clipper は CCW に揃える
    """
    if not subject or len(subject) < 3:
        return []
    if not clipper or len(clipper) < 3:
        return []

    clipper = ensure_ccw(clipper)
    output = subject[:]

    for i in range(len(clipper)):
        a = clipper[i]
        b = clipper[(i + 1) % len(clipper)]
        if not output:
            break

        input_list = output
        output = []

        prev = input_list[-1]
        prev_in = is_inside_halfplane_ccw(prev, a, b)

        for cur in input_list:
            cur_in = is_inside_halfplane_ccw(cur, a, b)

            if cur_in:
                if not prev_in:
                    output.append(line_intersection(prev, cur, a, b))
                output.append(cur)
            else:
                if prev_in:
                    output.append(line_intersection(prev, cur, a, b))

            prev, prev_in = cur, cur_in

    # 連続重複点の簡易除去
    cleaned: Poly = []
    for p in output:
        if not cleaned or (
            abs(p[0] - cleaned[-1][0]) > 1e-6 or abs(p[1] - cleaned[-1][1]) > 1e-6
        ):
            cleaned.append(p)
    if (
        len(cleaned) >= 2
        and abs(cleaned[0][0] - cleaned[-1][0]) < 1e-6
        and abs(cleaned[0][1] - cleaned[-1][1]) < 1e-6
    ):
        cleaned.pop()
    return cleaned


def poly_to_rotated_quad(poly: Poly) -> List[List[int]]:
    """
    任意poly -> 最小外接回転矩形(4点) (cv2.minAreaRect)
    TextDetectorSchema の points に入れたい場合に使う
    """
    if len(poly) == 0:
        return [[0, 0], [0, 0], [0, 0], [0, 0]]

    arr = np.array(poly, dtype=np.float32)
    rect = cv2.minAreaRect(arr)  # ((cx,cy),(w,h),angle)
    box = cv2.boxPoints(rect)  # 4x2
    box = box.astype(int).tolist()
    return box


def box_to_poly(box):
    x1, y1, x2, y2 = box
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def quad_to_poly(quad):
    # quad: [[x,y],[x,y],[x,y],[x,y]]
    return [tuple(p) for p in quad]


def split_text_poly_by_cells_poly(
    text_poly: Poly,
    cells: List[Dict[str, Any]],
    min_area_ratio: float = 0.05,
) -> List[Dict[str, Any]]:
    """
    text_poly: 文字領域poly（例：4点quad）
    cells: [{"id":..., "poly":[(x,y),...4点...], ...}, ...]  ※セルもpoly
    """
    base_area = poly_area(text_poly)
    if base_area <= 0:
        return []

    out = []
    for cell in cells:
        cell_poly = cell.get("poly")
        if not cell_poly or len(cell_poly) < 3:
            continue

        clipped = clip_convex_poly(text_poly, cell_poly)
        a = poly_area(clipped)
        if a <= 0:
            continue
        if a / base_area < min_area_ratio:
            continue

        out.append(
            {
                "cell_id": cell["id"],
                "poly": clipped,  # 交差poly（傾き保持）
                "area": a,
                "area_ratio": a / base_area,
            }
        )

    out.sort(key=lambda d: d["area"], reverse=True)
    return out


def normalize_quad_ccw(poly):
    # 重心
    cx = sum(p[0] for p in poly) / len(poly)
    cy = sum(p[1] for p in poly) / len(poly)

    # 角度でソート（反時計回り）
    poly = sorted(poly, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    return poly


def replace_spanning_words_with_clipped_polys_poly(
    words: List[Dict[str, Any]],
    cells: List[Dict[str, Any]],
    min_area_ratio=0.05,
    keep_unsplit=True,
):
    """
    words: [{"poly": [(x,y)..], "score":..., ...}, ...]   # poly(quad)前提
    cells: [{"id":..., "poly":[(x,y)..], ...}, ...]       # poly(quad)前提
    """
    new_words = []
    for w in words:
        wpoly = w["poly"]
        parts = split_text_poly_by_cells_poly(
            text_poly=wpoly,
            cells=cells,
            min_area_ratio=min_area_ratio,
        )

        if len(parts) == 0:
            new_words.append(w)  # テーブル外など
            continue

        if len(parts) == 1 and keep_unsplit:
            new_words.append(w)
            continue

        for p in parts:
            nw = dict(w)
            nw["poly"] = p["poly"]
            nw["area_ratio"] = p["area_ratio"]
            nw["cell_id"] = p["cell_id"]
            new_words.append(nw)

    return new_words


def build_text_detector_schema_from_split_words_rotated_quad(
    split_words: List[Dict[str, Any]],
    score_strategy="inherit",  # or "area_ratio"
):
    """
    split_words の poly を回転quadに戻して points に入れる
    """
    points = []
    scores = []

    for w in split_words:
        quad = poly_to_rotated_quad(w["poly"])  # 傾き保持
        points.append(normalize_quad_ccw(quad))

        if score_strategy == "area_ratio":
            base = w.get("score", 1.0)
            scores.append(base * w.get("area_ratio", 1.0))
        else:
            scores.append(w.get("score", 1.0))

    return {"points": points, "scores": scores}
