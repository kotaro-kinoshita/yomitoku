import cv2
import numpy as np
import torch


def load_image(image_path: str) -> np.ndarray:
    """
    Open an image file.

    Args:
        image_path (str): path to the image file

    Returns:
        np.ndarray: image data(BGR)
    """

    img = cv2.imread(image_path)
    validate_image(img)
    img = convert_3ch(img)
    return img


def validate_image(img: np.ndarray, minimum_thresh=32) -> None:
    """
    Validate the image data.

    Args:
        img (np.ndarray): image data
    """
    if img is None:
        raise ValueError("Image is not found.")

    h, w = img.shape[:2]
    if h < minimum_thresh or w < minimum_thresh:
        raise ValueError("Image size is too small.")


def convert_3ch(img: np.ndarray) -> np.ndarray:
    """
    Convert the image to RGB format
    Args:
        img (np.ndarray): target image

    Returns:
        np.ndarray: converted image
    """
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img


def resize_shortest_edge(
    img: np.ndarray, shortest_edge_length: int, max_length: int
) -> np.ndarray:
    """
    Resize the shortest edge of the image to `shortest_edge_length` while keeping the aspect ratio.
    if the longest edge is longer than `max_length`, resize the longest edge to `max_length` while keeping the aspect ratio.

    Args:
        img (np.ndarray): target image
        shortest_edge_length (int): pixel length of the shortest edge after resizing
        max_length (int): pixel length of maximum edge after resizing

    Returns:
        np.ndarray: resized image
    """

    h, w = img.shape[:2]
    scale = shortest_edge_length / min(h, w)
    if h < w:
        new_h, new_w = shortest_edge_length, int(w * scale)
    else:
        new_h, new_w = int(h * scale), shortest_edge_length

    if max(new_h, new_w) > max_length:
        scale = float(max_length) / max(new_h, new_w)
        new_h, new_w = int(new_h * scale), int(new_w * scale)

    neww = max(int(new_w / 32) * 32, 32)
    newh = max(int(new_h / 32) * 32, 32)

    img = cv2.resize(img, (neww, newh))
    return img


def normalize_image(
    img: np.ndarray, rgb=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
) -> np.ndarray:
    """
    Normalize the image data.

    Args:
        img (np.ndarray): target image

    Returns:
        np.ndarray: normalized image
    """
    img = img[:, :, ::-1].astype(np.float32)
    img = img / 255.0
    img = (img - np.array(rgb)) / np.array(std)

    return img


def array_to_tensor(img: np.ndarray) -> torch.Tensor:
    """
    Convert the image data to tensor.
    (H, W, C) -> (N, C, H, W)

    Args:
        img (np.ndarray): target image(H, W, C)

    Returns:
        torch.Tensor: (N, C, H, W) tensor
    """
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.as_tensor(img, dtype=torch.float)
    tensor = tensor[None, :, :, :]
    return tensor


def extract_roi_with_perspective(img, quad):
    """
    Extract the word image from the image with perspective transformation.

    Args:
        img (np.ndarray): target image
        polygon (np.ndarray): polygon vertices

    Returns:
        np.ndarray: extracted image
    """
    dst = img.copy()
    quad = np.array(quad, dtype=np.float32)
    width = np.linalg.norm(quad[0] - quad[1])
    height = np.linalg.norm(quad[1] - quad[2])

    width = int(width)
    height = int(height)

    pts1 = np.float32(quad)
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(dst, M, (width, height))

    return dst


def resize_with_padding(img, target_size, background_color=(0, 0, 0)):
    """
    Resize the image with padding.

    Args:
        img (np.ndarray): target image
        target_size (int, int): target size
        background_color (Tuple[int, int, int]): background color

    Returns:
        np.ndarray: resized image
    """
    h, w = img.shape[:2]

    if h > 3 * w:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h, w = img.shape[:2]

    scale_w = 1.0
    scale_h = 1.0
    if w > target_size[1]:
        scale_w = target_size[1] / w
    if h > target_size[0]:
        scale_h = target_size[0] / h

    new_w = int(w * min(scale_w, scale_h))
    new_h = int(h * min(scale_w, scale_h))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    canvas = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    canvas[:, :] = background_color

    resized_size = resized.shape[:2]
    canvas[: resized_size[0], : resized_size[1], :] = resized

    return canvas
