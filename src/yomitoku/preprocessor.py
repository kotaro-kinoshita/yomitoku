from yomitoku.dewarper import Dewarper
from yomitoku.page_dewarper import PageDewarper
from yomitoku.utils.logger import set_logger

logger = set_logger(__name__, "INFO")


class Preprocessor:
    """Image corrections before analysis: geometric correction (dewarping of curled, curved, tilted or keystoned
    pages). Same interface as yomitoku-pro's Preprocessor (which also corrects the rotation first).

    The dewarper is chosen by `configs["dewarper"]["model_name"]`: "pagenet_v1" (default) finds every page of the
    image (the two pages of a spread, receipts side by side, ...) and dewarps each; "uvdoc_v1" dewarps the image as
    one page.

    Args:
        dewarp_coordinate (str): "dewarped" (results are in the dewarped image, default) or "photo" (the caller maps
            them back to the input image with the page's `DewarpResult`).
    """

    def __init__(self, configs={}, device="cuda", dewarp_coordinate="dewarped"):
        dewarper_kwargs = {"device": device}

        if isinstance(configs, dict):
            if "dewarper" in configs:
                dewarper_kwargs.update(configs["dewarper"])
        else:
            raise ValueError("configs must be a dictionary")

        if dewarp_coordinate not in ("dewarped", "photo"):
            raise ValueError("dewarp_coordinate must be 'dewarped' or 'photo'")

        model_name = dewarper_kwargs.setdefault("model_name", "pagenet_v1")
        if model_name in PageDewarper.model_catalog.list_model():
            self.dewarper = PageDewarper(**dewarper_kwargs)
        else:
            self.dewarper = Dewarper(**dewarper_kwargs)
        self.dewarp_coordinate = dewarp_coordinate
        self.rotated_img = None  # input of the dewarper (named as in pro, where it is the rotation-corrected image)

    @property
    def multi_page(self):
        return isinstance(self.dewarper, PageDewarper)

    def pages(self, img):
        """Every page of the image: [(preprocess results {"dewarp": DewarpSchema}, page image, DewarpResult or None
        when not dewarped)]."""
        self.rotated_img = img
        if self.multi_page:
            outputs = self.dewarper(img, self.dewarp_coordinate)
            results = self.dewarper.last_results
        else:
            outputs = [self.dewarper(img, self.dewarp_coordinate)]
            results = [self.dewarper.last_result]
        return [
            ({"dewarp": schema}, page_img, results[k] if k < len(results) else None)
            for k, (schema, page_img) in enumerate(outputs)
        ]

    def run(self, img):
        """The first page only: (DewarpSchema, image)."""
        pages = self.pages(img)
        if len(pages) > 1:
            logger.warning(
                f"{len(pages)} pages found in the image; only the first is analyzed (DocumentAnalyzer.analyze_pages analyzes them all)"
            )
        pre, page_img, _ = pages[0]
        return pre["dewarp"], page_img

    def __call__(self, img):
        dewarp_outputs, corrected_img = self.run(img)
        return {"dewarp": dewarp_outputs}, corrected_img
