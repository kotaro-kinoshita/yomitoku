from .layout_parser import LayoutParser
from .table_structure_recognizer import TableStructureRecognizer

from .schemas import LayoutAnalyzerSchema
from .utils.dewarp import drop_off_paper_figures
from .utils.visualizer import layout_visualizer


class LayoutAnalyzer:
    def __init__(self, configs={}, device="cuda", visualize=False):
        layout_parser_kwargs = {
            "device": device,
            "visualize": visualize,
        }
        table_structure_recognizer_kwargs = {
            "device": device,
            "visualize": visualize,
        }

        if isinstance(configs, dict):
            if "layout_parser" in configs:
                layout_parser_kwargs.update(configs["layout_parser"])

            if "table_structure_recognizer" in configs:
                table_structure_recognizer_kwargs.update(
                    configs["table_structure_recognizer"]
                )
        else:
            raise ValueError(
                "configs must be a dict. See the https://kotaro-kinoshita.github.io/yomitoku-dev/usage/"
            )

        self.layout_parser = LayoutParser(
            **layout_parser_kwargs,
        )
        self.table_structure_recognizer = TableStructureRecognizer(
            **table_structure_recognizer_kwargs,
        )

    def _drop_off_paper_figures(self, img, layout_results, vis, paper):
        if paper is None:
            return layout_results, vis
        figures = drop_off_paper_figures(layout_results.figures, paper)
        if len(figures) == len(layout_results.figures):
            return layout_results, vis
        layout_results.figures = figures
        if vis is not None:
            vis = layout_visualizer(layout_results, img)
        return layout_results, vis

    def __call__(self, img, paper=None):
        """paper: for a dewarped image, its paper mask (`DewarpResult.paper`); figures off the paper or containing
        the whole page are dropped."""
        layout_results, vis = self.layout_parser(img)
        layout_results, vis = self._drop_off_paper_figures(
            img, layout_results, vis, paper
        )
        table_boxes = [table.box for table in layout_results.tables]
        table_results, vis = self.table_structure_recognizer(img, table_boxes, vis=vis)

        results = LayoutAnalyzerSchema(
            paragraphs=layout_results.paragraphs,
            tables=table_results,
            figures=layout_results.figures,
        )

        return results, vis
