import pypdfium2
import numpy as np
from ..schemas import OCRSchema

def is_close(bbox1, bbox2, threshold=0.5):
    """
    Check if two bounding boxes are close enough to be considered part of the same line/segment.
    bbox: [x1, y1, x2, y2] (left, top, right, bottom) - Image coordinates
    """
    # Calculate simple distance between edges
    # Horizontal text: check x distance and y overlap
    # Vertical text: check y distance and x overlap

    l1, t1, r1, b1 = bbox1
    l2, t2, r2, b2 = bbox2

    w1 = r1 - l1
    h1 = b1 - t1
    w2 = r2 - l2
    h2 = b2 - t2

    # Average size
    avg_w = (w1 + w2) / 2
    avg_h = (h1 + h2) / 2

    # Check for Horizontal grouping
    # Y overlap
    y_overlap = min(b1, b2) - max(t1, t2)
    # X distance (r1 to l2)
    x_dist = l2 - r1

    if y_overlap > 0.5 * min(h1, h2): # Significant vertical overlap
        if -0.2 * avg_w < x_dist < threshold * avg_w: # Close horizontally
             return "horizontal"

    # Check for Vertical grouping
    # X overlap
    x_overlap = min(r1, r2) - max(l1, l2)
    # Y distance (b1 to t2)
    y_dist = t2 - b1

    if x_overlap > 0.5 * min(w1, w2): # Significant horizontal overlap
        if -0.2 * avg_h < y_dist < threshold * avg_h: # Close vertically
             return "vertical"

    return None

def merge_bboxes(bboxes):
    """
    Merge a list of bboxes [x1, y1, x2, y2] into one.
    """
    bboxes = np.array(bboxes)
    l = np.min(bboxes[:, 0])
    t = np.min(bboxes[:, 1])
    r = np.max(bboxes[:, 2])
    b = np.max(bboxes[:, 3])
    return [l, t, r, b]

def bbox_to_quad(bbox):
    """
    Convert [l, t, r, b] to 4 points [[x,y], ...]
    """
    l, t, r, b = bbox
    return [[int(l), int(t)], [int(r), int(t)], [int(r), int(b)], [int(l), int(b)]]

def extract_text_from_pdf_page(page, scale=1.0):
    """
    Extracts text from a pypdfium2 page and groups it into 'words' compatible with OCRSchema.

    Args:
        page: pypdfium2.PdfPage
        scale: float, scaling factor to match the image resolution used in analysis

    Returns:
        list of dicts: [{'points': [[x,y]...], 'content': str, 'direction': str, ...}]
    """
    textpage = page.get_textpage()
    n_chars = textpage.count_chars()

    if n_chars == 0:
        return []

    page_height = page.get_height()

    chars_info = []

    # 1. Extract raw characters
    for i in range(n_chars):
        char = textpage.get_text_range(i, 1)
        if not char or char.isspace() or ord(char) < 32: # Skip control chars, basic spaces handled by logic
            continue

        # PDF coords: left, bottom, right, top (usually)
        # But pypdfium2 get_charbox docs say: left, bottom, right, top
        l, b, r, t = textpage.get_charbox(i)

        # Convert to Image Coords (Top-Left origin)
        # Image Y = (Page Height - PDF Y) * scale
        # Image X = PDF X * scale

        # Note: PDF Y is 0 at bottom.
        # img_top = (page_height - t) * scale  (t is higher Y value in PDF)
        # img_bottom = (page_height - b) * scale (b is lower Y value in PDF)

        img_l = l * scale
        img_r = r * scale
        img_t = (page_height - t) * scale
        img_b = (page_height - b) * scale

        # Sanity check for negative coords or swapped
        if img_l > img_r: img_l, img_r = img_r, img_l
        if img_t > img_b: img_t, img_b = img_b, img_t

        chars_info.append({
            "char": char,
            "bbox": [img_l, img_t, img_r, img_b]
        })

    if not chars_info:
        return []

    # 2. Group characters into lines/segments
    # Simple greedy clustering

    groups = [] # List of {'chars': [], 'bboxes': [], 'direction': None}

    # We iterate chars. Since PDF text extraction usually follows reading order roughly,
    # sequential checks might suffice for simple cases.

    current_group = {
        "chars": [chars_info[0]["char"]],
        "bboxes": [chars_info[0]["bbox"]],
        "direction": None
    }

    for i in range(1, len(chars_info)):
        prev = chars_info[i-1]
        curr = chars_info[i]

        rel = is_close(current_group["bboxes"][-1], curr["bbox"], threshold=1.0) # threshold relative to char size

        if rel:
            # Check if direction is consistent
            if current_group["direction"] is None:
                current_group["direction"] = rel
                current_group["chars"].append(curr["char"])
                current_group["bboxes"].append(curr["bbox"])
            elif current_group["direction"] == rel:
                current_group["chars"].append(curr["char"])
                current_group["bboxes"].append(curr["bbox"])
            else:
                # Direction changed (e.g. from horz line to next horz line via vertical jump? unlikely for single char)
                # But if we were building a horizontal line and suddenly found a char below...
                # Actually is_close checks for immediate adjacency.

                # If we were 'horizontal' and the next char is 'vertical' relation?
                # Usually means new line or weird layout.
                groups.append(current_group)
                current_group = {
                    "chars": [curr["char"]],
                    "bboxes": [curr["bbox"]],
                    "direction": None
                }
        else:
            # Not close, start new group
            groups.append(current_group)
            current_group = {
                "chars": [curr["char"]],
                "bboxes": [curr["bbox"]],
                "direction": None
            }

    groups.append(current_group)

    # 3. Format Output
    words = []
    for g in groups:
        if not g["chars"]: continue

        content = "".join(g["chars"])
        merged_bbox = merge_bboxes(g["bboxes"])
        points = bbox_to_quad(merged_bbox)
        direction = g["direction"] if g["direction"] else "horizontal" # Default to horizontal if single char

        words.append({
            "points": points,
            "content": content,
            "direction": direction,
            "det_score": 1.0, # Confidence 1.0 for PDF text
            "rec_score": 1.0
        })

    return words
