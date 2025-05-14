import cv2


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
        header_cols.append("_".join(header))

    table_dict = []
    for i in range(header_row, n_rows):
        row_dict = {}
        for j in range(n_cols):
            row_dict[header_cols[j]] = table_array[i][j]
        table_dict.append(row_dict)

    return table_dict
