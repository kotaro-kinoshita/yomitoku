from typing import List, Union

from pydantic import conlist, Field, model_serializer

from ..base import BaseSchema
from ..export import export_csv, export_html, export_markdown, export_json


class _DropUnsetDewarpFields:
    """`polygon` / `dewarp` / `preprocess` are set only with geometric correction (and `page` / `num_pages` / `score`
    only with the multi-page dewarper); when unset they are left out of model_dump() (and so of the JSON export),
    keeping the output of runs without it unchanged."""

    @model_serializer(mode="wrap")
    def _drop_unset_dewarp_fields(self, handler):
        d = handler(self)
        for k in ("polygon", "dewarp", "preprocess", "page", "num_pages", "score"):
            if k in d and d[k] is None:
                del d[k]
        return d


class Element(_DropUnsetDewarpFields, BaseSchema):
    id: Union[str, None] = Field(
        ...,
        description="Unique identifier of the element",
    )
    box: conlist(int, min_length=4, max_length=4) = Field(
        ...,
        description="Bounding box of the layout element in the format [x1, y1, x2, y2]",
    )
    polygon: Union[List[conlist(int, min_length=2, max_length=2)], None] = Field(
        None,
        description="Outline [[x, y], ...] in the photo before geometric correction, set only when the results were mapped back to it (box is then the outline's bounding box)",
    )
    score: float = Field(
        ...,
        description="Confidence score of the layout element detection",
    )
    role: Union[str, None] = Field(
        ...,
        description="Role of the element, e.g., ['section_headings', 'page_header', 'page_footer', 'list_item', 'caption', 'inline_formula', 'display_formula', 'index']",
    )
    contents: Union[str, None] = Field(
        ...,
        description="Text content of the element",
    )


class ParagraphSchema(_DropUnsetDewarpFields, BaseSchema):
    box: conlist(int, min_length=4, max_length=4) = Field(
        ...,
        description="Bounding box of the paragraph in the format [x1, y1, x2, y2]",
    )
    polygon: Union[List[conlist(int, min_length=2, max_length=2)], None] = Field(
        None,
        description="Outline [[x, y], ...] in the photo before geometric correction, set only when the results were mapped back to it (box is then the outline's bounding box)",
    )
    contents: Union[str, None] = Field(
        ...,
        description="Text content of the paragraph",
    )
    direction: Union[str, None] = Field(
        ...,
        description="Text direction, e.g., ['horizontal' or 'vertical']",
    )
    order: Union[int, None] = Field(
        ...,
        description="Order of the paragraph in the document",
    )
    role: Union[str, None] = Field(
        ...,
        description="Role of the paragraph, e.g., ['section_headings', 'page_header', 'page_footer'])",
    )


class TableCellSchema(_DropUnsetDewarpFields, BaseSchema):
    col: int = Field(
        ...,
        description="Column index of the cell",
    )
    row: int = Field(
        ...,
        description="Row index of the cell",
    )
    col_span: int = Field(
        ...,
        description="Number of columns spanned by the cell",
    )
    row_span: int = Field(
        ...,
        description="Number of rows spanned by the cell",
    )
    box: conlist(int, min_length=4, max_length=4) = Field(
        ...,
        description="Bounding box of the cell in the format [x1, y1, x2, y2]",
    )
    polygon: Union[List[conlist(int, min_length=2, max_length=2)], None] = Field(
        None,
        description="Outline [[x, y], ...] in the photo before geometric correction, set only when the results were mapped back to it (box is then the outline's bounding box)",
    )
    contents: Union[str, None] = Field(
        ...,
        description="Text content of the cell",
    )


class TableLineSchema(_DropUnsetDewarpFields, BaseSchema):
    box: conlist(int, min_length=4, max_length=4) = Field(
        ...,
        description="Bounding box of the table line in the format [x1, y1, x2, y2]",
    )
    polygon: Union[List[conlist(int, min_length=2, max_length=2)], None] = Field(
        None,
        description="Outline [[x, y], ...] in the photo before geometric correction, set only when the results were mapped back to it (box is then the outline's bounding box)",
    )
    score: float = Field(
        ...,
        description="Confidence score of the table line detection",
    )


class TableStructureRecognizerSchema(_DropUnsetDewarpFields, BaseSchema):
    box: conlist(int, min_length=4, max_length=4) = Field(
        ...,
        description="Bounding box of the table in the format [x1, y1, x2, y2]",
    )
    polygon: Union[List[conlist(int, min_length=2, max_length=2)], None] = Field(
        None,
        description="Outline [[x, y], ...] in the photo before geometric correction, set only when the results were mapped back to it (box is then the outline's bounding box)",
    )
    n_row: int = Field(..., description="Number of rows in the table")
    n_col: int = Field(..., description="Number of columns in the table")
    rows: List[TableLineSchema] = Field(
        ...,
        description="List of table lines representing rows",
    )
    cols: List[TableLineSchema] = Field(
        ...,
        description="List of table lines representing columns",
    )
    spans: List[TableLineSchema] = Field(
        ...,
        description="List of table lines representing spans",
    )
    cells: List[TableCellSchema] = Field(
        ...,
        description="List of table cells",
    )
    order: int = Field(
        ...,
        description="Order of the table in the document",
    )


class LayoutAnalyzerSchema(BaseSchema):
    paragraphs: List[Element] = Field(
        ...,
        description="List of detected paragraphs",
    )
    tables: List[TableStructureRecognizerSchema] = Field(
        ...,
        description="List of detected tables",
    )
    figures: List[Element] = Field(
        ...,
        description="List of detected figures",
    )


class WordPrediction(_DropUnsetDewarpFields, BaseSchema):
    points: conlist(
        conlist(int, min_length=2, max_length=2),
        min_length=4,
        max_length=4,
    ) = Field(
        ...,
        description="Bounding box of the word in the format [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]",
    )
    polygon: Union[List[conlist(int, min_length=2, max_length=2)], None] = Field(
        None,
        description="Outline [[x, y], ...] in the photo before geometric correction, set only when the results were mapped back to it (box is then the outline's bounding box)",
    )
    content: str = Field(..., description="Text content of the word")
    direction: str = Field(
        ..., description="Text direction, e.g., 'horizontal' or 'vertical'"
    )
    rec_score: float = Field(
        ..., description="Confidence score of the word recognition"
    )
    det_score: float = Field(
        ...,
        description="Confidence score of the word detection",
    )


class TextDetectorSchema(BaseSchema):
    points: List[
        conlist(
            conlist(int, min_length=2, max_length=2),
            min_length=4,
            max_length=4,
        )
    ] = Field(
        ...,
        description="List of bounding boxes of detected text regions in the format [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]",
    )
    scores: List[float] = Field(
        ...,
        description="List of confidence scores for each detected text region",
    )


class OCRSchema(BaseSchema):
    words: List[WordPrediction] = Field(
        ...,
        description="List of recognized words with their bounding boxes, content, direction, and scores",
    )


class LayoutParserSchema(BaseSchema):
    paragraphs: List[Element] = Field(..., description="List of detected paragraphs")
    tables: List[Element] = Field(..., description="List of detected tables")
    figures: List[Element] = Field(..., description="List of detected figures")


class FigureSchema(_DropUnsetDewarpFields, BaseSchema):
    box: conlist(int, min_length=4, max_length=4) = Field(
        ..., description="Bounding box of the figure in the format [x1, y1, x2, y2]"
    )
    polygon: Union[List[conlist(int, min_length=2, max_length=2)], None] = Field(
        None,
        description="Outline [[x, y], ...] in the photo before geometric correction, set only when the results were mapped back to it (box is then the outline's bounding box)",
    )
    order: Union[int, None] = Field(
        ..., description="Order of the figure in the document"
    )
    paragraphs: List[ParagraphSchema] = Field(
        ..., description="List of paragraphs associated with the figure"
    )
    order: Union[int, None] = Field(
        ..., description="Order of the figure in the document"
    )
    direction: Union[str, None] = Field(
        ..., description="Text direction, e.g., ['horizontal' or 'vertical']"
    )
    figure_path: Union[str, None] = Field(
        None,
        description="Relative path to the exported figure image from the output file",
    )


class DewarpSchema(_DropUnsetDewarpFields, BaseSchema):
    status: str = Field(
        ...,
        description="'ok': the page was dewarped; 'rejected': the predicted correction looked broken, the page was not dewarped (the image may need to be retaken)",
    )
    reasons: List[str] = Field(
        ..., description="Failed checks when rejected, e.g. ['mask_iou', 'spacing']"
    )
    coordinate: str = Field(
        ...,
        description="Coordinates of the results: 'dewarped' (the dewarped image) or 'photo' (the image before dewarping)",
    )
    size: Union[conlist(int, min_length=2, max_length=2), None] = Field(
        ..., description="[width, height] of the dewarped image (None when rejected)"
    )
    mask_iou: Union[float, None] = Field(
        ..., description="IoU of the predicted page grid and the predicted paper mask"
    )
    flipped_cells: float = Field(
        ..., description="Share of predicted grid cells turned over"
    )
    spacing_ratio: float = Field(
        ..., description="Longest / median predicted grid edge"
    )
    outside: float = Field(
        ..., description="Share of predicted grid nodes far outside the image"
    )
    page: Union[int, None] = Field(
        None,
        description="Multi-page dewarper: index of this page among the pages found in the image (left to right, then top to bottom)",
    )
    num_pages: Union[int, None] = Field(
        None,
        description="Multi-page dewarper: number of pages found in the image (spread, receipts side by side, ...)",
    )
    score: Union[float, None] = Field(
        None, description="Multi-page dewarper: page detection score"
    )


class PreprocessSchema(_DropUnsetDewarpFields, BaseSchema):
    dewarp: Union[DewarpSchema, None] = Field(
        None, description="Geometric correction result; None when it is disabled"
    )


class DocumentAnalyzerSchema(_DropUnsetDewarpFields, BaseSchema):
    preprocess: Union[PreprocessSchema, None] = Field(
        None,
        description="Preprocessing information of the document (geometric correction); None when disabled",
    )
    paragraphs: List[ParagraphSchema] = Field(
        ..., description="List of detected paragraphs"
    )
    tables: List[TableStructureRecognizerSchema] = Field(
        ..., description="List of detected tables"
    )
    words: List[WordPrediction] = Field(..., description="List of recognized words")
    figures: List[FigureSchema] = Field(..., description="List of detected figures")

    def to_html(self, out_path: str, **kwargs):
        return export_html(self, out_path, **kwargs)

    def to_markdown(self, out_path: str, **kwargs):
        return export_markdown(self, out_path, **kwargs)

    def to_csv(self, out_path: str, **kwargs):
        return export_csv(self, out_path, **kwargs)

    def to_json(self, out_path: str, **kwargs):
        return export_json(self, out_path, **kwargs)


class TextRecognizerSchema(BaseSchema):
    contents: List[str] = Field(
        ...,
        description="List of recognized text contents",
    )
    directions: List[str] = Field(
        ..., description="List of text directions, e.g., ['horizontal' or 'vertical']"
    )
    scores: List[float] = Field(
        ..., description="List of confidence scores for each recognized text"
    )
    points: List[
        conlist(
            conlist(int, min_length=2, max_length=2),
            min_length=4,
            max_length=4,
        )
    ] = Field(
        ...,
        description="List of bounding boxes of recognized text in the format [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]",
    )
