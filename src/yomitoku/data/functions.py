from pathlib import Path

import cv2
import numpy as np
import torch
import pypdfium2

from ..constants import (
    MIN_IMAGE_SIZE,
    SUPPORT_INPUT_FORMAT,
    WARNING_IMAGE_SIZE,
)
from ..utils.logger import set_logger

logger = set_logger(__name__)


def load_image(image_path: str) -> np.ndarray:
    """
    Open an image file.

    Args:
        image_path (str): path to the image file

    Returns:
        np.ndarray: image data(BGR)
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"File not found: {image_path}")

    ext = image_path.suffix[1:].lower()
    if ext not in SUPPORT_INPUT_FORMAT:
        raise ValueError(
            f"Unsupported image format. Supported formats are {SUPPORT_INPUT_FORMAT}"
        )

    if ext == "pdf":
        raise ValueError(
            "PDF file is not supported by load_image(). Use load_pdf() instead."
        )

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image data.")

    h, w = img.shape[:2]
    if h < MIN_IMAGE_SIZE or w < MIN_IMAGE_SIZE:
        raise ValueError("Image size is too small.")

    if min(h, w) < WARNING_IMAGE_SIZE:
        logger.warning(
            """
            The image size is small, which may result in reduced OCR accuracy. 
            The process will continue, but it is recommended to input images with a minimum size of 720 pixels on the shorter side.
            """
        )

    return img


def load_pdf(pdf_path: str, dpi=200) -> list[np.ndarray]:
    """
    Open a PDF file.

    Args:
        pdf_path (str): path to the PDF file

    Returns:
        list[np.ndarray]: list of image data(BGR)
    """

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"File not found: {pdf_path}")

    ext = pdf_path.suffix[1:].lower()
    if ext not in SUPPORT_INPUT_FORMAT:
        raise ValueError(
            f"Unsupported image format. Supported formats are {SUPPORT_INPUT_FORMAT}"
        )

    if ext != "pdf":
        raise ValueError(
            "image file is not supported by load_pdf(). Use load_image() instead."
        )

    try:
        doc = pypdfium2.PdfDocument(pdf_path)
        renderer = doc.render(
            pypdfium2.PdfBitmap.to_pil,
            scale=dpi / 72,
        )
        images = list(renderer)
        images = [np.array(image.convert("RGB"))[:, :, ::-1] for image in images]

        doc.close()
    except Exception as e:
        raise ValueError(f"Failed to open the PDF file: {pdf_path}") from e

    return images


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

    img = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_AREA)
    return img


def standardization_image(
    img: np.ndarray, rgb=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
) -> np.ndarray:
    """
    Normalize the image data.

    Args:
        img (np.ndarray): target image

    Returns:
        np.ndarray: normalized image
    """
    img = img[:, :, ::-1]
    img = img / 255.0
    img = (img - np.array(rgb)) / np.array(std)
    img = img.astype(np.float32)

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


def validate_quads(img: np.ndarray, quads: list[list[list[int]]]):
    """
    Validate the vertices of the quadrilateral.

    Args:
        img (np.ndarray): target image
        quads (list[list[list[int]]]): list of quadrilateral

    Raises:
        ValueError: if the vertices are invalid
    """

    h, w = img.shape[:2]
    for quad in quads:
        if len(quad) != 4:
            raise ValueError("The number of vertices must be 4.")

        for point in quad:
            if len(point) != 2:
                raise ValueError("The number of coordinates must be 2.")

        quad = np.array(quad, dtype=int)
        x1 = np.min(quad[:, 0])
        x2 = np.max(quad[:, 0])
        y1 = np.min(quad[:, 1])
        y2 = np.max(quad[:, 1])
        h, w = img.shape[:2]

        if x1 < 0 or x2 > w or y1 < 0 or y2 > h:
            raise ValueError(f"The vertices are out of the image. {quad.tolist()}")

    return True


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


def rotate_text_image(img, thresh_aspect=2):
    """
    Rotate the image if the aspect ratio is too high.

    Args:
        img (np.ndarray): target image
        thresh_aspect (int): threshold of aspect ratio

    Returns:
        np.ndarray: rotated image
    """
    h, w = img.shape[:2]
    if h > thresh_aspect * w:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


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
    scale_w = 1.0
    scale_h = 1.0
    if w > target_size[1]:
        scale_w = target_size[1] / w
    if h > target_size[0]:
        scale_h = target_size[0] / h

    new_w = int(w * min(scale_w, scale_h))
    new_h = int(h * min(scale_w, scale_h))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    canvas[:, :] = background_color

    resized_size = resized.shape[:2]
    canvas[: resized_size[0], : resized_size[1], :] = resized

    return canvas
