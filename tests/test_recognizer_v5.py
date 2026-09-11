import inspect
from unittest.mock import Mock

import numpy as np
import pytest
from omegaconf import OmegaConf

from yomitoku.base import BaseModule
from yomitoku.configs import TextRecognizerPARSeqTinyDynwV5Config
from yomitoku.data.dataset import ParseqDataset
from yomitoku.data.functions import (
    calc_resize_without_padding,
    resize_with_dynamic_padding,
)
from yomitoku.text_recognizer import TextRecognizer


def test_resize_policies():
    crop = np.full((8, 20, 3), 255, dtype=np.uint8)
    assert calc_resize_without_padding(crop, (32, 800), "fit") == (32, 80)
    assert calc_resize_without_padding(crop, (32, 800), "downscale") == (8, 20)
    assert resize_with_dynamic_padding(crop, (32, 800), resize_policy="fit").shape == (
        32,
        144,
        3,
    )
    assert resize_with_dynamic_padding(
        crop, (32, 800), resize_policy="downscale"
    ).shape == (32, 88, 3)
    with pytest.raises(ValueError):
        calc_resize_without_padding(crop, (32, 800), "invalid")


def test_dataset_uses_config_and_explicit_override():
    cfg = OmegaConf.structured(TextRecognizerPARSeqTinyDynwV5Config)
    img = np.full((16, 48, 3), 255, dtype=np.uint8)
    quads = [[[1, 1], [30, 1], [30, 9], [1, 9]]]
    kwargs = (
        {"det_scores": [1.0]}
        if "det_scores" in inspect.signature(ParseqDataset).parameters
        else {}
    )
    dynamic = ParseqDataset(cfg, img, quads, **kwargs)
    fixed = ParseqDataset(cfg, img, quads, dynamic_width=False, **kwargs)
    assert dynamic.data[0].shape[1] < 800
    assert fixed.data[0].shape[1] == 800
    assert dynamic.content_widths == fixed.content_widths


@pytest.mark.parametrize(
    "override,expected", [(None, True), (False, False), (True, True)]
)
def test_recognizer_runtime_resolves_config(monkeypatch, override, expected):
    cfg = OmegaConf.structured(TextRecognizerPARSeqTinyDynwV5Config)
    monkeypatch.setattr(BaseModule, "__init__", lambda self, **kwargs: None)

    def load(self, *args, **kwargs):
        self._cfg = cfg
        self.model = Mock()

    monkeypatch.setattr(TextRecognizer, "load_model", load)
    # Exercise constructor resolution independently of pro singleton caching.
    rec = object.__new__(TextRecognizer)
    TextRecognizer.__init__(
        rec, device="cpu", dynamic_width=override, batch_bucketing=override
    )
    assert rec.dynamic_width is expected
    assert rec.batch_bucketing is expected
    assert rec.nfkc_normalize is False
    assert "℡".translate(rec.char_replace_table) == "TEL"


@pytest.mark.parametrize(
    "lite,explicit", [(False, None), (True, None), (False, "tiny"), (True, "old")]
)
def test_cli_selects_v5_without_forcing_runtime(monkeypatch, tmp_path, lite, explicit):
    import importlib
    import sys

    cli = importlib.import_module("yomitoku.cli.main")
    default = (
        inspect.signature(TextRecognizer.__init__).parameters["model_name"].default
    )
    public = default == "parseq-middle-dynw-v5"
    tiny = "parseq-tiny-dynw-v5" if public else "parseqv5-tiny-dynw"
    old = "parseq-large-v4_1" if public else "parseqv4"
    selected = {"tiny": tiny, "old": old}.get(explicit)
    expected = selected or (tiny if lite else default)
    input_path = tmp_path / "input.png"
    input_path.touch()
    argv = ["yomitoku", str(input_path)]
    if lite:
        argv.append("-l")
    if selected:
        argv += ["--tr_name", selected]
    monkeypatch.setattr(sys, "argv", argv)

    class Captured(Exception):
        pass

    def analyzer(**kwargs):
        recognizer = kwargs["configs"]["ocr"]["text_recognizer"]
        assert recognizer["model_name"] == expected
        assert "dynamic_width" not in recognizer
        assert "batch_bucketing" not in recognizer
        raise Captured

    monkeypatch.setattr(cli, "DocumentAnalyzer", analyzer)
    with pytest.raises(Captured):
        cli.main()
