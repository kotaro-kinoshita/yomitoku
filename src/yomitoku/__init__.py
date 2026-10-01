from importlib.metadata import version

from .document_analyzer import DocumentAnalyzer
from .layout_analyzer import LayoutAnalyzer
from .layout_parser import LayoutParser
from .ocr import OCR
from .studio_form_template import (
    StudioFormTemplate,
    StudioFormTemplateParser,
    apply_studio_form_template,
    load_studio_form_template,
)
from .table_semantic_parser import TableSemanticParser
from .table_structure_recognizer import TableStructureRecognizer
from .text_detector import TextDetector
from .text_recognizer import TextRecognizer

__all__ = [
    "OCR",
    "LayoutParser",
    "TableStructureRecognizer",
    "TableSemanticParser",
    "StudioFormTemplate",
    "StudioFormTemplateParser",
    "load_studio_form_template",
    "apply_studio_form_template",
    "TextDetector",
    "TextRecognizer",
    "LayoutAnalyzer",
    "DocumentAnalyzer",
]
__version__ = version(__package__)
