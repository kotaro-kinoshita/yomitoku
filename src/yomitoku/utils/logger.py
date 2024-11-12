import warnings
from logging import Formatter, StreamHandler, getLogger


def set_logger(name, level="INFO"):
    logger = getLogger(name)
    logger.setLevel(level)
    handler = StreamHandler()
    handler.setLevel(level)
    format = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(format)
    logger.addHandler(handler)

    warnings.filterwarnings("ignore")
    return logger
