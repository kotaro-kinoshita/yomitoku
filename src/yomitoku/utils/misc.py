import cv2
import math

from typing import List, Tuple, Dict, Any

Point = Tuple[float, float]
Poly = List[Point]
Rect = Tuple[float, float, float, float]  # x1,y1,x2,y2


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


def _clip_poly_halfplane(poly: Poly, inside_fn, intersect_fn) -> Poly:
    """Sutherland–Hodgman: poly を half-plane でクリップ"""
    if not poly:
        return []
    out: Poly = []
    prev = poly[-1]
    prev_in = inside_fn(prev)

    for cur in poly:
        cur_in = inside_fn(cur)
        if cur_in:
            if not prev_in:
                out.append(intersect_fn(prev, cur))
            out.append(cur)
        else:
            if prev_in:
                out.append(intersect_fn(prev, cur))
        prev, prev_in = cur, cur_in

    return out


def _intersect_with_vertical(p1: Point, p2: Point, x: float) -> Point:
    """線分(p1->p2)と垂直線 x=const の交点（p1.x!=p2.x想定）"""
    x1, y1 = p1
    x2, y2 = p2
    if x2 == x1:
        return (x, y1)  # 退避（実際は平行）
    t = (x - x1) / (x2 - x1)
    return (x, y1 + t * (y2 - y1))


def _intersect_with_horizontal(p1: Point, p2: Point, y: float) -> Point:
    """線分(p1->p2)と水平線 y=const の交点（p1.y!=p2.y想定）"""
    x1, y1 = p1
    x2, y2 = p2
    if y2 == y1:
        return (x1, y)  # 退避（実際は平行）
    t = (y - y1) / (y2 - y1)
    return (x1 + t * (x2 - x1), y)


def clip_poly_to_rect(poly: Poly, rect: Rect) -> Poly:
    """任意ポリゴンを軸平行矩形でクリップして返す"""
    x1, y1, x2, y2 = rect
    # Left: x >= x1
    poly = _clip_poly_halfplane(
        poly,
        inside_fn=lambda p: p[0] >= x1,
        intersect_fn=lambda a, b: _intersect_with_vertical(a, b, x1),
    )
    # Right: x <= x2
    poly = _clip_poly_halfplane(
        poly,
        inside_fn=lambda p: p[0] <= x2,
        intersect_fn=lambda a, b: _intersect_with_vertical(a, b, x2),
    )
    # Top: y >= y1
    poly = _clip_poly_halfplane(
        poly,
        inside_fn=lambda p: p[1] >= y1,
        intersect_fn=lambda a, b: _intersect_with_horizontal(a, b, y1),
    )
    # Bottom: y <= y2
    poly = _clip_poly_halfplane(
        poly,
        inside_fn=lambda p: p[1] <= y2,
        intersect_fn=lambda a, b: _intersect_with_horizontal(a, b, y2),
    )
    return poly


def poly_area(poly: Poly) -> float:
    """符号なし面積（Shoelace）"""
    if len(poly) < 3:
        return 0.0
    s = 0.0
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + [poly[0]]):
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def poly_to_quad(poly):
    """
    任意多角形 poly -> 軸平行 quad [[x1,y1]...[x4,y4]]
    """
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

    return [
        [int(x1), int(y1)],
        [int(x2), int(y1)],
        [int(x2), int(y2)],
        [int(x1), int(y2)],
    ]


def poly_to_aabb(poly: Poly) -> Rect:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (min(xs), min(ys), max(xs), max(ys))


def split_text_quad_by_cells(
    text_quad: Poly,
    cells: List[Dict[str, Any]],
    min_area_ratio: float = 0.05,
) -> List[Dict[str, Any]]:
    """
    text_quad: 4点のquad（順序は時計回り/反時計回りどちらでもOKだが自己交差はNG）
    cells: [{"id":..., "box":[x1,y1,x2,y2], ...}, ...]
    min_area_ratio: 元quad面積に対して小さすぎる欠片を捨てる閾値
    return: [{"cell_id":..., "poly":..., "aabb":..., "area":...}, ...]
    """
    base_area = poly_area(text_quad)
    if base_area <= 0:
        return []

    out = []
    for cell in cells:
        rect = tuple(cell["box"])
        clipped = clip_poly_to_rect(text_quad, rect)
        a = poly_area(clipped)
        if a <= 0:
            continue
        if a / base_area < min_area_ratio:
            continue
        out.append(
            {
                "cell_id": cell["id"],
                "poly": clipped,  # セル内に切られたポリゴン（分割片）
                "aabb": poly_to_aabb(clipped),  # 使いやすいように軸平行bboxも添付
                "area": a,
                "area_ratio": a / base_area,
            }
        )

    # 面積が大きい順に
    out.sort(key=lambda d: d["area"], reverse=True)
    return out


def replace_spanning_words_with_clipped_polys(
    words,
    cells,
    min_area_ratio=0.05,
    keep_unsplit=True,
):
    """
    words: list[dict] 例 {"id": "w1", "poly": [(x,y),...], "text": "..."}
    cells: list[dict] 例 {"id": "c1", "box": [x1,y1,x2,y2]}
    min_area_ratio: 小さすぎる欠片を捨てる
    keep_unsplit: 1セルしかヒットしない場合に元wordを残すか

    return: new_words
    """

    new_words = []
    for w in words:
        parts = split_text_quad_by_cells(
            text_quad=w,
            cells=cells,
            min_area_ratio=min_area_ratio,
        )

        # ヒットなし → そのまま（テーブル外とか）
        if len(parts) == 0:
            new_words.append(
                {
                    "poly": w,
                }
            )
            continue

        # 1セルだけ → 置換しない（任意）
        if len(parts) == 1 and keep_unsplit:
            new_words.append(
                {
                    "poly": w,
                }
            )
            continue

        for k, p in enumerate(parts):
            # nw = {"poly": p["poly"], "aabb": p["aabb"], "cell_id": p["cell_id"]}
            new_words.append(
                {
                    "poly": p["poly"],
                }
            )

    # print(new_words)

    return new_words


def build_text_detector_schema_from_split_words(
    split_words,
    score_strategy="inherit",  # or "area_ratio"
):
    """
    split_words: replace_spanning_words_with_clipped_polys の出力
    return: dict compatible with TextDetectorSchema
    """

    points = []
    scores = []

    for w in split_words:
        quad = poly_to_quad(w["poly"])
        points.append(quad)

        if score_strategy == "area_ratio":
            base_score = w.get("score", 1.0)
            scores.append(base_score * w.get("area_ratio", 1.0))
        else:
            scores.append(w.get("score", 1.0))

    return {
        "points": points,
        "scores": scores,
    }
