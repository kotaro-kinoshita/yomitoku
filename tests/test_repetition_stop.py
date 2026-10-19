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
    assert cfg.rep_min_run_p1 == 16
    assert cfg.rep_min_repeats == 3
    # A --tr_cfg YAML can switch it off for any model.
    assert cfg.rec_orientation_fallback is False
    assert cfg.rec_orientation_fallback_thresh == 0.75
    # A --tr_cfg YAML can override them for any model.
    merged = OmegaConf.merge(
        cfg,
        OmegaConf.create({"repetition_stop": False, "rec_orientation_fallback": True}),
    )
    assert merged.repetition_stop is False
    assert merged.rec_orientation_fallback is True


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


@pytest.mark.parametrize(
    "cfg_values,kwargs,expected",
    [
        ((False, 0.75), {}, (False, 0.75)),
        ((True, 0.6), {}, (True, 0.6)),
        (
            (True, 0.6),
            {"rec_orientation_fallback": False, "rec_orientation_fallback_thresh": 0.9},
            (False, 0.9),
        ),
    ],
)
def test_orientation_fallback_argument_overrides_config(
    monkeypatch, cfg_values, kwargs, expected
):
    config_class, _ = CATALOG.get(MODEL_NAMES[0])
    cfg = OmegaConf.structured(config_class)
    cfg.rec_orientation_fallback, cfg.rec_orientation_fallback_thresh = cfg_values
    monkeypatch.setattr(BaseModule, "__init__", lambda self, **kwargs: None)

    def load(self, *args, **kwargs):
        self._cfg = cfg
        self.model = Mock()

    monkeypatch.setattr(TextRecognizer, "load_model", load)
    rec = object.__new__(TextRecognizer)
    TextRecognizer.__init__(rec, device="cpu", **kwargs)
    assert (rec.rec_orientation_fallback, rec.rec_orientation_fallback_thresh) == (
        expected
    )


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


@pytest.mark.parametrize(
    "args,expected",
    [
        ([], {}),
        (["--disable-repetition-stop"], {"repetition_stop": False}),
        (["--enable-rec-orientation-fallback"], {"rec_orientation_fallback": True}),
        (
            [
                "--enable-rec-orientation-fallback",
                "--rec-orientation-fallback-thresh",
                "0.6",
            ],
            {"rec_orientation_fallback": True, "rec_orientation_fallback_thresh": 0.6},
        ),
    ],
)
def test_cli_recognizer_options(monkeypatch, tmp_path, args, expected):
    import importlib

    cli = importlib.import_module("yomitoku.cli.main")
    input_path = tmp_path / "input.png"
    input_path.touch()
    argv = ["yomitoku", str(input_path), *args]
    monkeypatch.setattr(sys, "argv", argv)

    class Captured(Exception):
        pass

    def analyzer(**kwargs):
        recognizer = kwargs["configs"]["ocr"]["text_recognizer"]
        keys = (
            "repetition_stop",
            "rec_orientation_fallback",
            "rec_orientation_fallback_thresh",
        )
        assert {k: recognizer[k] for k in keys if k in recognizer} == expected
        raise Captured

    monkeypatch.setattr(cli, "DocumentAnalyzer", analyzer)
    with pytest.raises(Captured):
        cli.main()
