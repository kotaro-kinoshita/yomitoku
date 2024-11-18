import pytest
import torch
from omegaconf import OmegaConf

from yomitoku.ocr import OCR


def test_ocr():
    device = "cpu"
    visualize = True
    config = {
        "text_detector": {
            "path_cfg": "tests/yaml/text_detector.yaml",
        },
        "text_recognizer": {
            "path_cfg": "tests/yaml/text_recognizer.yaml",
        },
    }

    ocr = OCR(configs=config, device=device, visualize=visualize)

    assert ocr.detector.device == torch.device(device)
    assert ocr.recognizer.device == torch.device(device)
    assert ocr.detector.visualize == visualize
    assert ocr.recognizer.visualize == visualize

    text_detector_cfg = OmegaConf.load(config["text_detector"]["path_cfg"])
    text_recognizer_cfg = OmegaConf.load(config["text_recognizer"]["path_cfg"])

    assert (
        ocr.detector.post_processor.thresh
        == text_detector_cfg.post_process.thresh
    )

    assert (
        ocr.recognizer.model.refine_iters == text_recognizer_cfg.refine_iters
    )


def test_ocr_invalid_path():
    config = {
        "text_detector": {
            "path_cfg": "tests/yaml/dummy.yaml",
        },
    }

    with pytest.raises(FileNotFoundError):
        OCR(configs=config)


def test_ocr_invalid_config():
    with pytest.raises(ValueError):
        OCR(configs="invalid")

    config = {"test": "invalid"}
    with pytest.raises(AssertionError):
        OCR(configs=config)
