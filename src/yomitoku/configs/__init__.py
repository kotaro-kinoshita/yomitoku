from .cfg_layout_parser_rtdtrv2 import LayoutParserRTDETRv2Config
from .cfg_layout_parser_rtdtrv2_v2 import LayoutParserRTDETRv2V2Config
from .cfg_table_structure_recognizer_rtdtrv2 import (
    TableStructureRecognizerRTDETRv2Config,
)
from .cfg_text_detector_dbnet import TextDetectorDBNetConfig
from .cfg_text_detector_dbnet_v2 import TextDetectorDBNetV2Config
from .cfg_text_recognizer_parseq import TextRecognizerPARSeqConfig
from .cfg_text_recognizer_parseq_small import TextRecognizerPARSeqSmallConfig
from .cfg_text_recognizer_parseq_tiny import TextRecognizerPARSeqTinyConfig
from .cfg_text_recognizer_parseq_v2 import TextRecognizerPARSeqV2Config
from .cfg_table_cell_parser_rtdtrv2_beta import TableCellParserRTDETRv2BetaConfig

DEFAULT_CONFIGS = [
    TextRecognizerPARSeqV2Config,
    TextDetectorDBNetV2Config,
    LayoutParserRTDETRv2V2Config,
    TableStructureRecognizerRTDETRv2Config,
    TableCellParserRTDETRv2BetaConfig,
]

__all__ = [
    "TextDetectorDBNetConfig",
    "TextRecognizerPARSeqConfig",
    "LayoutParserRTDETRv2Config",
    "TextRecognizerPARSeqTinyConfig",
    "TableStructureRecognizerRTDETRv2Config",
    "TextRecognizerPARSeqSmallConfig",
    "LayoutParserRTDETRv2V2Config",
    "TextDetectorDBNetV2Config",
    "TextRecognizerPARSeqV2Config",
    "TableCellParserRTDETRv2BetaConfig",
]
