from importlib.metadata import version

from .ocr import OCR

__all__ = [
    "OCR",
]
__version__ = version(__package__)
