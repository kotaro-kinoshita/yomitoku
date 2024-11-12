import time

import torch
from omegaconf import OmegaConf

from yomitoku.utils.logger import set_logger

from .configs import load_config

logger = set_logger(__name__, "INFO")


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


class BaseModule:
    def __init__(self, *args, **kwds):
        pass

    def __new__(cls, *args, **kwds):
        logger.info(f"Initialize {cls.__name__}")
        cls.__call__ = observer(cls, cls.__call__)
        return super().__new__(cls)

    def __call__(self, *args, **kwds):
        pass

    def set_config(self, path_cfg):
        self._cfg = load_config(self.__class__.__name__, path_cfg)

    def save_config(self, path_cfg):
        OmegaConf.save(self._cfg, path_cfg)

    def log_config(self):
        logger.info(OmegaConf.to_yaml(self._cfg))

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
