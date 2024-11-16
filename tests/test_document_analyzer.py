import pytest
import torch
from omegaconf import OmegaConf

from yomitoku import DocumentAnalyzer


def test_initialize():
    device = "cpu"
    visualize = True
    config = {
        "ocr": {
            "text_detector": {
                "path_cfg": "tests/yaml/text_detector.yaml",
            },
            "text_recognizer": {
                "path_cfg": "tests/yaml/text_recognizer.yaml",
            },
        },
        "layout_analyzer": {
            "layout_parser": {
                "path_cfg": "tests/yaml/layout_parser.yaml",
            },
            "table_structure_recognizer": {
                "path_cfg": "tests/yaml/table_structure_recognizer.yaml",
            },
        },
    }

    analyzer = DocumentAnalyzer(
        configs=config, device=device, visualize=visualize
    )

    # サブモジュールのパラメータが更新されているか確認
    assert analyzer.ocr.detector.device == torch.device(device)
    assert analyzer.ocr.recognizer.device == torch.device(device)
    assert analyzer.layout.layout_parser.device == torch.device(device)
    assert analyzer.layout.table_structure_recognizer.device == torch.device(
        device
    )

    assert analyzer.ocr.detector.visualize == visualize
    assert analyzer.ocr.recognizer.visualize == visualize
    assert analyzer.layout.layout_parser.visualize == visualize
    assert analyzer.layout.table_structure_recognizer.visualize == visualize

    text_detector_cfg = OmegaConf.load(
        config["ocr"]["text_detector"]["path_cfg"]
    )
    text_recognizer_cfg = OmegaConf.load(
        config["ocr"]["text_recognizer"]["path_cfg"]
    )
    layout_parser_cfg = OmegaConf.load(
        config["layout_analyzer"]["layout_parser"]["path_cfg"]
    )
    table_structure_recognizer_cfg = OmegaConf.load(
        config["layout_analyzer"]["table_structure_recognizer"]["path_cfg"]
    )

    assert (
        analyzer.ocr.detector.post_processor.thresh
        == text_detector_cfg.post_process.thresh
    )

    assert (
        analyzer.ocr.recognizer.model.refine_iters
        == text_recognizer_cfg.refine_iters
    )

    assert (
        analyzer.layout.layout_parser.thresh_score
        == layout_parser_cfg.thresh_score
    )

    assert (
        analyzer.layout.table_structure_recognizer.thresh_score
        == table_structure_recognizer_cfg.thresh_score
    )


def test_invalid_path():
    config = {
        "ocr": {
            "text_detector": {
                "path_cfg": "tests/yaml/dummy.yaml",
            },
        }
    }

    with pytest.raises(FileNotFoundError):
        DocumentAnalyzer(
            configs=config,
        )


def test_invalid_config():
    with pytest.raises(ValueError):
        DocumentAnalyzer(
            configs="invalid",
        )
