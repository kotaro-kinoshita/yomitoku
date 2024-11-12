from omegaconf import OmegaConf

from . import (
    LayoutParserConfig,
    TableStructureRecognizerConfig,
    TextDetectorConfig,
    TextRecognizerConfig,
)


def get_default_config(module_name):
    if module_name == "TextDetector":
        return TextDetectorConfig
    if module_name == "TextRecognizer":
        return TextRecognizerConfig
    if module_name == "LayoutParser":
        return LayoutParserConfig
    if module_name == "TableStructureRecognizer":
        return TableStructureRecognizerConfig
    raise ValueError(f"Unknown module: {module_name}")


def load_yaml_config(path_config):
    if not path_config.exists():
        raise FileNotFoundError(f"Config file not found: {path_config}")

    with open(path_config, "r") as file:
        yaml_config = OmegaConf.load(file)
    return yaml_config


def load_config(module_name, path_config=None):
    cfg = OmegaConf.structured(get_default_config(module_name))
    if path_config is not None:
        yaml_config = load_yaml_config(path_config)
        cfg = OmegaConf.merge(cfg, yaml_config)
    return cfg
