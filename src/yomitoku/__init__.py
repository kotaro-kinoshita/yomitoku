from importlib.metadata import version

from .layout_parser import LayoutParser
from .ocr import OCR
from .table_structure_recognizer import TableStructureRecognizer

__all__ = [
    "OCR",
    "LayoutParser",
    "TableStructureRecognizer",
]
__version__ = version(__package__)
