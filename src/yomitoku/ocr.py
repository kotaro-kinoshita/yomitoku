from yomitoku.base import BaseModule
from yomitoku.text_detector import TextDetector
from yomitoku.text_recognitizer import TextRecognizer


class OCR:
    def __init__(self, det_cfg, rec_cfg, device, visualize=True):
        self.detector = TextDetector(
            path_cfg=det_cfg, visualize=visualize, device=device
        )
        self.recognizer = TextRecognizer(
            path_cfg=rec_cfg, visualize=visualize, device=device
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

    def __call__(self, img):
        """_summary_

        Args:
            img (np.ndarray): cv2 image(BGR)
        """

        h, w = img.shape[:2]
        det_outputs, vis = self.detector(img)
        rec_outputs, vis = self.recognizer(img, det_outputs["quads"], vis=vis)

        results = {}
        results.update(image_size=(w, h))
        results.update(words=self.format(det_outputs, rec_outputs))
        return results, vis
