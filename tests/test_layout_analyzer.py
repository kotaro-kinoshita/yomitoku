import pytest
import torch
from omegaconf import OmegaConf

from yomitoku.layout_analyzer import LayoutAnalyzer


def test_layout():
    device = "cpu"
    visualize = True
    config = {
        "layout_parser": {
            "path_cfg": "tests/yaml/layout_parser.yaml",
        },
        "table_structure_recognizer": {
            "path_cfg": "tests/yaml/table_structure_recognizer.yaml",
        },
    }

    analyzer = LayoutAnalyzer(configs=config, device=device, visualize=visualize)

    assert analyzer.layout_parser.device == torch.device(device)
    assert analyzer.table_structure_recognizer.device == torch.device(device)

    assert analyzer.layout_parser.visualize == visualize
    assert analyzer.table_structure_recognizer.visualize == visualize

    layout_parser_cfg = OmegaConf.load(config["layout_parser"]["path_cfg"])
    table_structure_recognizer_cfg = OmegaConf.load(
        config["table_structure_recognizer"]["path_cfg"]
    )

    assert analyzer.layout_parser.thresh_score == layout_parser_cfg.thresh_score

    assert (
        analyzer.table_structure_recognizer.thresh_score
        == table_structure_recognizer_cfg.thresh_score
    )


def test_layout_invalid_path():
    config = {
        "layout_parser": {
            "path_cfg": "tests/yaml/dummy.yaml",
        },
    }
    with pytest.raises(FileNotFoundError):
        LayoutAnalyzer(configs=config)
