from typing import List

from pydantic import conlist

from yomitoku.text_detector import TextDetector
from yomitoku.text_recognizer import TextRecognizer

from .base import BaseSchema


class WordPrediction(BaseSchema):
    points: conlist(
        conlist(int, min_length=2, max_length=2),
        min_length=4,
        max_length=4,
    )
    content: str
    direction: str
    det_score: float
    rec_score: float


class OCRSchema(BaseSchema):
    words: List[WordPrediction]


class OCR:
    def __init__(self, configs=None, device="cuda", visualize=False):
        text_detector_kwargs = {
            "device": device,
            "visualize": visualize,
        }
        text_recognizer_kwargs = {
            "device": device,
            "visualize": visualize,
        }

        if "text_detector" in configs:
            text_detector_kwargs.update(configs["text_detector"])
        if "text_recognizer" in configs:
            text_recognizer_kwargs.update(configs["text_recognizer"])

        self.detector = TextDetector(**text_detector_kwargs)
        self.recognizer = TextRecognizer(**text_recognizer_kwargs)

    def aggregate(self, det_outputs, rec_outputs):
        words = []
        for points, det_score, pred, rec_score, direction in zip(
            det_outputs.points,
            det_outputs.scores,
            rec_outputs.contents,
            rec_outputs.scores,
            rec_outputs.directions,
        ):
            words.append(
                {
                    "points": points,
                    "content": pred,
                    "direction": direction,
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

        det_outputs, vis = self.detector(img)
        rec_outputs, vis = self.recognizer(img, det_outputs.points, vis=vis)

        outputs = {"words": self.aggregate(det_outputs, rec_outputs)}
        results = OCRSchema(**outputs)
        return results, vis
