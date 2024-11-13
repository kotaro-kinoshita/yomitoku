import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union

from pydantic import BaseModel, conlist

from .data.functions import load_image
from .layout_analyzer import LayoutAnalyzer
from .ocr import OCR, WordPrediction
from .table_structure_recognizer import TableStructureRecognizerSchema
from .utils.export import export_html
from .utils.misc import is_contained, quad_to_xyxy


class ParagraphSchema(BaseModel):
    box: conlist(int, min_length=4, max_length=4)
    contents: Union[str, None]
    direction: Union[str, None]


class DocumentAnalyzerSchema(BaseModel):
    paragraphs: List[ParagraphSchema]
    tables: List[TableStructureRecognizerSchema]
    words: List[WordPrediction]


class DocumentAnalyzer:
    def __init__(self, configs=None, device="cpu", visualize=True):
        if configs is None:
            configs = {
                "ocr": {
                    "text_detector": {"path_cfg": None, "model_name": "dbnet"},
                    "text_recognizer": {
                        "path_cfg": None,
                        "model_name": "parseq",
                    },
                },
                "layout_alalyzer": {
                    "layout_parser": {
                        "model_name": "rtdetrv2",
                        "path_cfg": None,
                    },
                    "table_structure_recognizer": {
                        "model_name": "rtdetrv2",
                        "path_cfg": None,
                    },
                },
            }

        self.ocr = OCR(
            configs=configs["ocr"],
            visualize=visualize,
        )

        self.layout = LayoutAnalyzer(
            configs=configs["layout_alalyzer"],
            visualize=visualize,
        )

    def aggregate(self, ocr_res, layout_res):
        paragraphs = []
        check_list = [False] * len(ocr_res.words)
        for table in layout_res.table:
            for cell in table.cells:
                words = []

                word_sum_width = 0
                word_sum_height = 0
                for i, word in enumerate(ocr_res.words):
                    word_box = quad_to_xyxy(word.points)
                    if is_contained(cell.box, word_box, threshold=0.6):
                        words.append(word)
                        word_sum_width += word_box[2] - word_box[0]
                        word_sum_height += word_box[3] - word_box[1]
                        check_list[i] = True

                if len(words) == 0:
                    continue

                mean_width = word_sum_width / len(words)
                mean_height = word_sum_height / len(words)

                direction = "horizontal"
                if mean_height > mean_width:
                    direction = "vertical"

                if direction == "horizontal":
                    words = sorted(
                        words,
                        key=lambda x: (
                            x.points[0][1] // int(mean_height),
                            x.points[0][0],
                        ),
                    )
                else:
                    words = sorted(
                        words,
                        key=lambda x: (
                            x.points[1][0] // int(mean_width),
                            x.points[1][1],
                        ),
                        reverse=True,
                    )

                words = "\n".join([content.content for content in words])
                cell.contents = words

        paragraph_boxes = [paragraph.box for paragraph in layout_res.paragraph]
        for paragraph_box in paragraph_boxes:
            words = []

            word_sum_width = 0
            word_sum_height = 0
            for i, word in enumerate(ocr_res.words):
                word_box = quad_to_xyxy(word.points)
                if is_contained(paragraph_box, word_box):
                    words.append(word)
                    word_sum_width += word_box[2] - word_box[0]
                    word_sum_height += word_box[3] - word_box[1]
                    check_list[i] = True

            if len(words) == 0:
                continue

            mean_width = word_sum_width / len(words)
            mean_height = word_sum_height / len(words)

            direction = "horizontal"
            if mean_height > mean_width:
                direction = "vertical"

            if mean_width > mean_height:
                words = sorted(
                    words,
                    key=lambda x: (
                        x.points[0][1] // int(mean_height),
                        x.points[0][0],
                    ),
                )
            else:
                words = sorted(
                    words,
                    key=lambda x: (
                        x.points[1][0] // int(mean_width),
                        x.points[1][1],
                    ),
                    reverse=True,
                )

            words = "\n".join([content.content for content in words])
            paragraph = {
                "contents": words,
                "box": paragraph_box,
                "direction": direction,
            }

            paragraph = ParagraphSchema(**paragraph)
            paragraphs.append(paragraph)

        for i, word in enumerate(ocr_res.words):
            direction = "horizontal"
            height = word.points[1][1] - word.points[0][1]
            width = word.points[1][0] - word.points[0][0]

            if 3 * height > width:
                direction = "vertical"

            if not check_list[i]:
                paragraph = {
                    "contents": word.content,
                    "box": quad_to_xyxy(word.points),
                    "direction": direction,
                }

                paragraph = ParagraphSchema(**paragraph)
                paragraphs.append(paragraph)

        outputs = {
            "paragraphs": paragraphs,
            "tables": layout_res.table,
            "words": ocr_res.words,
        }

        return outputs

    async def __call__(self, img):
        with ThreadPoolExecutor(max_workers=2) as executor:
            loop = asyncio.get_running_loop()
            tasks = [
                loop.run_in_executor(executor, self.ocr, img),
                loop.run_in_executor(executor, self.layout, img),
            ]

            results = await asyncio.gather(*tasks)

            results_ocr, _ = results[0]
            results_layout, _ = results[1]

        outputs = self.aggregate(results_ocr, results_layout)
        results = DocumentAnalyzerSchema(**outputs)
        return results


if __name__ == "__main__":
    analyzer = DocumentAnalyzer()
    img = "dataset/test/00001890_4013504_24.jpg"
    img = load_image(img)
    results = asyncio.run(analyzer(img))
    export_html(results)
