from typing import List, Union
from pydantic import conlist, Field

from ..base import BaseSchema
from ..export import (
    export_csv,
    export_html,
    export_markdown,
    export_json,
)


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


class Element(BaseSchema):
    box: conlist(int, min_length=4, max_length=4) = Field(
        ...,
        description="Bounding box of the layout element in the format [x1, y1, x2, y2]",
    )
    score: float = Field(
        ...,
        description="Confidence score of the layout element detection",
    )
    role: Union[str, None] = Field(
        ...,
        description="Role of the element, e.g., ['section_headings', 'page_header', 'page_footer', 'list_item', 'caption', 'inline_formula', 'display_formula', 'index']",
    )
    order: Union[int, None] = Field(
        ...,
        description="Order of the element in the document",
    )


class ParagraphSchema(BaseSchema):
    box: conlist(int, min_length=4, max_length=4) = Field(
        ...,
        description="Bounding box of the paragraph in the format [x1, y1, x2, y2]",
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


class TableLineSchema(BaseSchema):
    box: conlist(int, min_length=4, max_length=4) = Field(
        ...,
        description="Bounding box of the table line in the format [x1, y1, x2, y2]",
    )
    score: float = Field(
        ...,
        description="Confidence score of the table line detection",
    )


class TableCellSchema(BaseSchema):
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
    contents: Union[str, None] = Field(
        ...,
        description="Text content of the cell",
    )


class TableStructureRecognizerSchema(BaseSchema):
    box: conlist(int, min_length=4, max_length=4) = Field(
        ...,
        description="Bounding box of the table in the format [x1, y1, x2, y2]",
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


class WordPrediction(BaseSchema):
    points: conlist(
        conlist(int, min_length=2, max_length=2),
        min_length=4,
        max_length=4,
    ) = Field(
        ...,
        description="Bounding box of the word in the format [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]",
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


class FigureSchema(BaseSchema):
    box: conlist(int, min_length=4, max_length=4) = Field(
        ..., description="Bounding box of the figure in the format [x1, y1, x2, y2]"
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


class DocumentAnalyzerSchema(BaseSchema):
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
