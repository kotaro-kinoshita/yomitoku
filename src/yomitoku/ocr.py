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
    rec_score: float
    det_score: float


class OCRSchema(BaseSchema):
    words: List[WordPrediction]


def ocr_aggregate(det_outputs, rec_outputs):
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


class OCR:
    def __init__(self, configs={}, device="cuda", visualize=False):
        text_detector_kwargs = {
            "device": device,
            "visualize": visualize,
        }
        text_recognizer_kwargs = {
            "device": device,
            "visualize": visualize,
        }

        if isinstance(configs, dict):
            if "text_detector" in configs:
                text_detector_kwargs.update(configs["text_detector"])
            if "text_recognizer" in configs:
                text_recognizer_kwargs.update(configs["text_recognizer"])
        else:
            raise ValueError(
                "configs must be a dict. See the https://kotaro-kinoshita.github.io/yomitoku-dev/usage/"
            )

        self.detector = TextDetector(**text_detector_kwargs)
        self.recognizer = TextRecognizer(**text_recognizer_kwargs)

    def __call__(self, imgs):
        """_summary_

        Args:
            img (np.ndarray): cv2 image(BGR)
        """

        det_outputs, vis_imgs = self.detector(imgs)

        points = [det_output.points for det_output in det_outputs]
        rec_outputs, vis_imgs = self.recognizer(imgs, points, vis_imgs=vis_imgs)

        results = []
        for det_output, rec_output in zip(det_outputs, rec_outputs):
            output = {"words": ocr_aggregate(det_output, rec_output)}
            result = OCRSchema(**output)
            results.append(result)

        return results, vis_imgs
