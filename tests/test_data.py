import numpy as np
import pytest

from yomitoku.data.functions import (
    array_to_tensor,
    load_image,
    load_pdf,
    resize_shortest_edge,
    resize_with_padding,
    rotate_text_image,
    standardization_image,
    validate_quads,
)


def test_load_image():
    with pytest.raises(FileNotFoundError):
        load_image("dummy.jpg")

    with pytest.raises(ValueError):
        load_image("tests/data/test.txt")

    with pytest.raises(ValueError):
        load_image("tests/data/small.jpg")

    with pytest.raises(ValueError):
        load_image("tests/data/test.pdf")

    with pytest.raises(ValueError):
        load_image("tests/data/invalid.jpg")

    targets = [
        "tests/data/test.jpg",
        "tests/data/test.png",
        "tests/data/test.tiff",
        "tests/data/test.bmp",
        "tests/data/test_gray.jpg",
        "tests/data/rgba.png",
        "tests/data/sampldoc.tif",
    ]

    for target in targets:
        image = load_image(target)
        assert len(image) >= 1
        assert image[0].shape[2] == 3
        assert image[0].shape[0] > 32
        assert image[0].shape[1] > 32
        assert image[0].dtype == "uint8"


def test_load_pdf():
    with pytest.raises(FileNotFoundError):
        load_pdf("dummy.pdf")

    with pytest.raises(ValueError):
        load_pdf("tests/data/test.txt")

    with pytest.raises(ValueError):
        load_pdf("tests/data/invalid.pdf")

    targets = [
        "tests/data/test.jpg",
        "tests/data/test.png",
        "tests/data/test.tiff",
        "tests/data/test.bmp",
        "tests/data/test_gray.jpg",
    ]

    for target in targets:
        with pytest.raises(ValueError):
            load_pdf(target)

    target = "tests/data/test.pdf"
    images = load_pdf(target)
    assert len(images) == 2
    for image in images:
        assert image.shape[2] == 3
        assert image.shape[0] > 0
        assert image.shape[1] > 0
        assert image.dtype == "uint8"


def test_resize_shortest_edge():
    img = np.zeros((1920, 1920, 3), dtype=np.uint8)
    resized = resize_shortest_edge(img, 1280, 1500)
    h, w = resized.shape[:2]
    assert min(h, w) == 1280
    assert h % 32 == 0
    assert w % 32 == 0

    img = np.zeros((1280, 1920, 3), dtype=np.uint8)
    resized = resize_shortest_edge(img, 1280, 1600)
    h, w = resized.shape[:2]
    assert max(h, w) == 1600
    assert h % 32 == 0
    assert w % 32 == 0

    resized = resize_shortest_edge(img, 1000, 1000)
    h, w = resized.shape[:2]
    assert h % 32 == 0
    assert w % 32 == 0


def test_standardization_image():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    normalized = standardization_image(img)
    assert normalized.shape == img.shape
    assert normalized.dtype == "float32"


def test_array_to_tensor():
    img = np.random.randint(0, 255, (100, 50, 3), dtype=np.uint8)
    tensor = array_to_tensor(img)
    assert tensor.shape == (1, 3, 100, 50)


def test_rotate_image():
    img = np.random.randint(0, 255, (100, 30, 3), dtype=np.uint8)
    rotated = rotate_text_image(img, thresh_aspect=2)
    assert rotated.shape == (30, 100, 3)

    img = np.random.randint(0, 255, (30, 100, 3), dtype=np.uint8)
    rotated = rotate_text_image(img, thresh_aspect=2)
    assert rotated.shape == (30, 100, 3)


def test_resize_with_padding():
    img = np.random.randint(0, 255, (50, 100, 3), dtype=np.uint8)
    resized = resize_with_padding(img, (50, 100))
    assert resized.shape == (50, 100, 3)

    img = np.random.randint(0, 255, (50, 150, 3), dtype=np.uint8)
    resized = resize_with_padding(img, (50, 100))
    assert resized.shape == (50, 100, 3)

    img = np.random.randint(0, 255, (60, 100, 3), dtype=np.uint8)
    resized = resize_with_padding(img, (50, 100))
    assert resized.shape == (50, 100, 3)


def test_validate_quads():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    quad = [[0, 0], [0, 10], [10, 10]]

    assert validate_quads(img, quad) is None

    quad = [[0], [0, 10], [10, 10], [10, 0]]

    assert validate_quads(img, quad) is None

    quad = [[0, 0], [0, 150], [10, 150], [10, 0]]

    assert validate_quads(img, quad) is None

    quad = [[150, 0], [150, 10], [10, 10], [10, 0]]

    assert validate_quads(img, quad) is None

    quad = [[-1, 0], [-1, 10], [10, 10], [10, 0]]

    assert validate_quads(img, quad) is None

    quad = [[0, -1], [0, 10], [10, 10], [10, -1]]

    assert validate_quads(img, quad) is None

    quads = [
        [[0, 0], [0, 10], [10, 10], [10, 0]],
        [[0, 0], [0, 20], [10, 20], [10, 0]],
        [[10, 0], [10, 30], [80, 30], [80, 0]],
    ]

    for quad in quads:
        assert validate_quads(img, quad)
