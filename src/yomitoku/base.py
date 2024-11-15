import time

import torch
from omegaconf import OmegaConf
from pydantic import BaseModel, Extra

from .export import export_json
from .utils.logger import set_logger

logger = set_logger(__name__, "INFO")


def load_yaml_config(path_config):
    if not path_config.exists():
        raise FileNotFoundError(f"Config file not found: {path_config}")

    with open(path_config, "r") as file:
        yaml_config = OmegaConf.load(file)
    return yaml_config


def load_config(
    default_config,
    path_config=None,
):
    cfg = OmegaConf.structured(default_config)
    if path_config is not None:
        yaml_config = load_yaml_config(path_config)
        cfg = OmegaConf.merge(cfg, yaml_config)
    return cfg


def observer(cls, func):
    def wrapper(*args, **kwargs):
        try:
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logger.info(
                f"{cls.__name__} {func.__name__} elapsed_time: {elapsed}"
            )
        except Exception as e:
            logger.error(
                f"Error occurred in {cls.__name__} {func.__name__}: {e}"
            )
            raise e
        return result

    return wrapper


class BaseSchema(BaseModel):
    class Config:
        extra = Extra.forbid
        validate_assignment = True

    def to_json(self, out_path: str):
        export_json(self, out_path)


class BaseModule:
    model_catalog = None

    def __init__(self):
        if self.model_catalog is None:
            raise NotImplementedError

        if not issubclass(self.model_catalog.__class__, BaseModelCatalog):
            raise ValueError(
                f"{self.model_catalog.__class__} is not SubClass BaseModelCatalog."
            )

        if len(self.model_catalog.list_model()) == 0:
            raise ValueError("No model is registered.")

    def __new__(cls, *args, **kwds):
        logger.info(f"Initialize {cls.__name__}")
        cls.__call__ = observer(cls, cls.__call__)
        return super().__new__(cls)

    def load_model(self, name, path_cfg):
        default_cfg, Net = self.model_catalog.get(name)
        self._cfg = load_config(default_cfg, path_cfg)
        self.model = Net.from_pretrained(self._cfg.hf_hub_repo, cfg=self._cfg)

    def save_config(self, path_cfg):
        OmegaConf.save(self._cfg, path_cfg)

    def log_config(self):
        logger.info(OmegaConf.to_yaml(self._cfg))

    @classmethod
    def catalog(cls):
        display = ""
        for model in cls.model_catalog.list_model():
            display += f"{model} "
        logger.info(f"{cls.__name__} Implemented Models")
        logger.info(display)

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        if "cuda" in device:
            if torch.cuda.is_available():
                self._device = torch.device(device)
            else:
                self._device = torch.device("cpu")
                logger.warning("CUDA is not available. Use CPU instead.")
        else:
            self._device = torch.device("cpu")


class BaseModelCatalog:
    def __init__(self):
        self.catalog = {}

    def get(self, model_name):
        model_name = model_name.lower()
        if model_name in self.catalog:
            return self.catalog[model_name]

        raise ValueError(f"Unknown model: {model_name}")

    def register(self, model_name, config, model):
        if model_name in self.catalog:
            raise ValueError(f"{model_name} is already registered.")

        self.catalog[model_name] = (config, model)

    def list_model(self):
        return list(self.catalog.keys())
