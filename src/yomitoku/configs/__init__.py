from .cfg_layout_parser_rtdtrv2 import LayoutParserRTDETRv2Config
from .cfg_layout_parser_rtdtrv2_v2 import LayoutParserRTDETRv2V2Config
from .cfg_table_structure_recognizer_rtdtrv2 import (
    TableStructureRecognizerRTDETRv2Config,
)
from .cfg_text_detector_dbnet import TextDetectorDBNetConfig
from .cfg_text_detector_dbnet_v2 import TextDetectorDBNetV2Config
from .cfg_text_detector_dbnet_v2_1 import TextDetectorDBNetV2_1Config
from .cfg_text_recognizer_parseq import TextRecognizerPARSeqConfig
from .cfg_text_recognizer_parseq_small import TextRecognizerPARSeqSmallConfig
from .cfg_text_recognizer_parseq_tiny import TextRecognizerPARSeqTinyConfig
from .cfg_text_recognizer_parseq_v2 import TextRecognizerPARSeqV2Config
from .cfg_text_recognizer_parseq_large_v4_1 import TextRecognizerPARSeqLargeV41Config
from .cfg_table_cell_parser_rtdtrv2_beta import TableCellParserRTDETRv2BetaConfig

DEFAULT_CONFIGS = [
    TextRecognizerPARSeqLargeV41Config,
    TextDetectorDBNetV2_1Config,
    LayoutParserRTDETRv2V2Config,
    TableStructureRecognizerRTDETRv2Config,
    TableCellParserRTDETRv2BetaConfig,
]

__all__ = [
    "TextDetectorDBNetConfig",
    "TextDetectorDBNetV2Config",
    "TextDetectorDBNetV2_1Config",
    "TextRecognizerPARSeqConfig",
    "TextRecognizerPARSeqTinyConfig",
    "TextRecognizerPARSeqSmallConfig",
    "TextRecognizerPARSeqV2Config",
    "TextRecognizerPARSeqLargeV41Config",
    "LayoutParserRTDETRv2Config",
    "LayoutParserRTDETRv2V2Config",
    "TableStructureRecognizerRTDETRv2Config",
    "TableCellParserRTDETRv2BetaConfig",
]
