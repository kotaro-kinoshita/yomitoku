from unittest.mock import patch

import pytest

from yomitoku.base import (
    BaseModelCatalog,
    BaseModule,
    load_config,
    load_yaml_config,
)
from yomitoku.configs import LayoutParserRTDETRv2Config
from yomitoku.models import RTDETRv2


def test_load_yaml_config():
    path_cfg = "tests/yaml/dummy.yaml"

    with pytest.raises(FileNotFoundError):
        load_yaml_config(path_cfg)

    with pytest.raises(ValueError):
        load_yaml_config("tests/data/test.jpg")

    path_cfg = "tests/yaml/layout_parser.yaml"
    yaml_config = load_yaml_config(path_cfg)
    assert yaml_config.thresh_score == 0.8


def test_load_config():
    default_config = LayoutParserRTDETRv2Config
    path_config = "tests/yaml/dummy.yaml"
    path_config = "tests/yaml/layout_parser.yaml"
    cfg = load_config(default_config, path_config)
    assert cfg.thresh_score == 0.8
    assert (
        cfg.hf_hub_repo
        == "KotaroKinoshita/yomitoku-layout-parser-rtdtrv2-open-beta"
    )


class TestModelCatalog(BaseModelCatalog):
    def __init__(self):
        super().__init__()
        self.register(
            "test",
            LayoutParserRTDETRv2Config,
            RTDETRv2,
        )


class TestModule(BaseModule):
    model_catalog = TestModelCatalog()

    def __init__(self):
        super().__init__()

    def __call__(self):
        pass


def test_base_model(tmp_path):
    module = TestModule()
    module.load_model("test", None)
    assert isinstance(module.model, RTDETRv2)

    module.save_config(tmp_path / "config.yaml")
    data = load_yaml_config(tmp_path / "config.yaml")
    default = LayoutParserRTDETRv2Config()
    assert data.hf_hub_repo == default.hf_hub_repo

    module.log_config()
    module.catalog()


def test_base_catalog():
    catalog = TestModelCatalog()
    assert catalog.list_model() == ["test"]

    with pytest.raises(ValueError):
        catalog.get("dummy")

    list = catalog.list_model()
    assert list == ["test"]

    with pytest.raises(ValueError):
        catalog.register("test", None, None)


def test_base_call():
    with patch("yomitoku.base.observer") as mock:
        module = TestModule()
        module()
        mock.assert_called_once()


def test_invalid_base_model():
    class InvalidModel(BaseModule):
        def __init__(self):
            super().__init__()

    with pytest.raises(NotImplementedError):
        InvalidModel()

    class InvalidModel(BaseModule):
        model_catalog = 1

    with pytest.raises(ValueError):
        InvalidModel()

    class InvalidModelCatalog(BaseModelCatalog):
        def __init__(self):
            self.catalog = {}

    class InvalidModel(BaseModule):
        model_catalog = InvalidModelCatalog()

    with pytest.raises(ValueError):
        InvalidModel()
