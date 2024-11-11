from yomitoku.utils.logger import set_logger
import time

logger = set_logger(__name__)


def observer(cls, func):
    def wrapper(*args, **kwargs):
        try:
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logger.info(f"{cls.__name__} {func.__name__} elapsed_time: {elapsed}")
        except Exception as e:
            logger.error(f"Error occurred in {cls.__name__} {func.__name__}: {e}")
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
