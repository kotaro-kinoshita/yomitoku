def load_charset(charset_path):
    with open(charset_path, "r") as f:
        charset = f.read()
    return charset


def filter_by_flag(elements, flags):
    assert len(elements) == len(flags)
    return [element for element, flag in zip(elements, flags) if flag]


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

    intersection = calc_intersection(rect_a, rect_b)
    if intersection is None:
        return False

    ix1, iy1, ix2, iy2 = intersection

    overlap_width = ix2 - ix1
    overlap_height = iy2 - iy1
    bx1, by1, bx2, by2 = rect_b

    b_area = (bx2 - bx1) * (by2 - by1)
    overlap_area = overlap_width * overlap_height

    if overlap_area / b_area > threshold:
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


def is_intersected_horizontal(rect_a, rect_b):
    _, ay1, _, ay2 = map(int, rect_a)
    _, by1, _, by2 = map(int, rect_b)

    # 交差領域の左上と右下の座標
    iy1 = max(ay1, by1)
    iy2 = min(ay2, by2)

    overlap_height = max(0, iy2 - iy1)

    if overlap_height == 0:
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
