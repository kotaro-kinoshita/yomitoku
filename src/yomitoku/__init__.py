from importlib.metadata import version

from .document_analyzer import DocumentAnalyzer
from .layout_analyzer import LayoutAnalyzer
from .layout_parser import LayoutParser
from .ocr import OCR
from .table_structure_recognizer import TableStructureRecognizer
from .text_detector import TextDetector
from .text_recognizer import TextRecognizer
from .table_detector import TableDetector
from .cell_detector import CellDetector

__all__ = [
    "OCR",
    "LayoutParser",
    "TableDetector",
    "CellDetector",
    "TableStructureRecognizer",
    "TextDetector",
    "TextRecognizer",
    "LayoutAnalyzer",
    "DocumentAnalyzer",
]
__version__ = version(__package__)
