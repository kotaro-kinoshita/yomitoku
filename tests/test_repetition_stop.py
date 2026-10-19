import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from yomitoku.base import BaseModule
from yomitoku.models.parseq import PARSeq
from yomitoku.text_recognizer import TextRecognizer, TextRecognizerModelCatalog

CATALOG = TextRecognizerModelCatalog()
MODEL_NAMES = CATALOG.list_model()


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_every_recognizer_config_declares_repetition_stop(model_name):
    config_class, _ = CATALOG.get(model_name)
    cfg = OmegaConf.structured(config_class)
    assert cfg.repetition_stop is True
    assert cfg.rep_period_max == 8
    assert cfg.rep_min_run_p1 in (8, 16)
    assert cfg.rep_min_repeats == 3
    # A --tr_cfg YAML can switch it off for any model.
    merged = OmegaConf.merge(cfg, OmegaConf.create({"repetition_stop": False}))
    assert merged.repetition_stop is False


@pytest.mark.parametrize(
    "cfg_value,override,expected",
    [
        (True, None, True),
        (False, None, False),
        (True, False, False),
        (False, True, True),
    ],
)
def test_recognizer_argument_overrides_config(
    monkeypatch, cfg_value, override, expected
):
    config_class, _ = CATALOG.get(MODEL_NAMES[0])
    cfg = OmegaConf.structured(config_class)
    cfg.repetition_stop = cfg_value
    monkeypatch.setattr(BaseModule, "__init__", lambda self, **kwargs: None)

    def load(self, *args, **kwargs):
        self._cfg = cfg
        self.model = Mock()

    monkeypatch.setattr(TextRecognizer, "load_model", load)
    rec = object.__new__(TextRecognizer)
    TextRecognizer.__init__(rec, device="cpu", repetition_stop=override)
    assert rec.repetition_stop is expected
    assert rec.model.repetition_stop is expected


def _detector(p1=8, repeats=3):
    return SimpleNamespace(rep_period_max=8, rep_min_run_p1=p1, rep_min_repeats=repeats)


def test_detect_repeat_onset():
    detect = PARSeq._detect_repeat_onset
    # period-1 run: needs rep_min_run_p1 copies
    assert detect(_detector(), [1, 2] + [5] * 7) is None
    assert detect(_detector(), [1, 2] + [5] * 8) == (2, 1)
    # multi-token unit: needs rep_min_repeats copies
    assert detect(_detector(), [9] + [3, 4] * 2) is None
    assert detect(_detector(), [9] + [3, 4] * 3) == (1, 2)


@pytest.mark.parametrize("flag", [False, True])
def test_cli_disable_repetition_stop(monkeypatch, tmp_path, flag):
    import importlib

    cli = importlib.import_module("yomitoku.cli.main")
    input_path = tmp_path / "input.png"
    input_path.touch()
    argv = ["yomitoku", str(input_path)]
    if flag:
        argv.append("--disable-repetition-stop")
    monkeypatch.setattr(sys, "argv", argv)

    class Captured(Exception):
        pass

    def analyzer(**kwargs):
        recognizer = kwargs["configs"]["ocr"]["text_recognizer"]
        if flag:
            assert recognizer["repetition_stop"] is False
        else:
            assert "repetition_stop" not in recognizer
        raise Captured

    monkeypatch.setattr(cli, "DocumentAnalyzer", analyzer)
    with pytest.raises(Captured):
        cli.main()
