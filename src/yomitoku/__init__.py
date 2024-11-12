from importlib.metadata import version

from .layout_parser import LayoutParser
from .ocr import OCR

__all__ = [
    "OCR",
    "LayoutParser",
]
__version__ = version(__package__)
