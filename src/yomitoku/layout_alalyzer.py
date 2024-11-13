from .layout_parser import LayoutParser
from .table_structure_recognizer import TableStructureRecognizer


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
        layout_preds, vis = self.layout_parser(img)
        table_boxes = layout_preds["table"]["boxes"]
        table_preds, vis = self.table_structure_recognizer(
            img, table_boxes, vis=vis
        )

        layout_preds["table"] = table_preds
        return layout_preds, vis
