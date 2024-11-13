from typing import List

from pydantic import BaseModel, conlist

from .layout_parser import Element, LayoutParser
from .table_structure_recognizer import (
    TableStructureRecognizer,
    TableStructureRecognizerSchema,
)


class LayoutAnalyzerSchema(BaseModel):
    paragraph: List[Element]
    table: List[TableStructureRecognizerSchema]
    figure: List[Element]


class LayoutAnalyzer:
    def __init__(self, configs=None, device="cpu", visualize=True):
        if configs is None:
            configs = {
                "layout_parser": {
                    "model_name": "rtdetrv2",
                    "path_cfg": None,
                },
                "table_structure_recognizer": {
                    "model_name": "rtdetrv2",
                    "path_cfg": None,
                },
            }

        self.layout_parser = LayoutParser(
            **configs["layout_parser"],
            device=device,
            visualize=visualize,
        )
        self.table_structure_recognizer = TableStructureRecognizer(
            **configs["table_structure_recognizer"],
            device=device,
            visualize=visualize,
        )

    def __call__(self, img):
        layout_results, vis = self.layout_parser(img)
        table_boxes = [table.box for table in layout_results.table]
        table_results, vis = self.table_structure_recognizer(
            img, table_boxes, vis=vis
        )

        results = LayoutAnalyzerSchema(
            paragraph=layout_results.paragraph,
            table=table_results,
            figure=layout_results.figure,
        )

        return results, vis
