from .data.functions import load_image
from .modules import BaseModule, Detection, Recognition
from .utils.misc import load_config


class OCR(BaseModule):
    def __init__(self, path_cfg, visualize=True):
        self.cfg = load_config(path_cfg)
        self.detector = Detection(
            self.cfg, visualize=visualize, device=self.cfg.DEVICE
        )
        self.recognizer = Recognition(
            self.cfg, visualize=visualize, device=self.cfg.DEVICE
        )

    def format(self, det_outputs, rec_outputs):
        words = []
        for quad, det_score, pred, rec_score in zip(
            det_outputs["quads"],
            det_outputs["scores"],
            rec_outputs["contents"],
            rec_outputs["scores"],
        ):
            words.append(
                {
                    "points": quad,
                    "content": pred,
                    "det_score": det_score,
                    "rec_score": rec_score,
                }
            )
        return words

    def __call__(self, path_img):
        img = load_image(path_img)
        h, w = img.shape[:2]
        det_outputs, vis = self.detector(img)
        rec_outputs, vis = self.recognizer(img, det_outputs["quads"], vis=vis)

        results = {}
        results.update(image_size=(w, h))
        results.update(words=self.format(det_outputs, rec_outputs))
        return results, vis
