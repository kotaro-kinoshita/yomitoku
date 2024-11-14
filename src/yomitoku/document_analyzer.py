import asyncio
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union

import cv2
from pydantic import BaseModel, conlist

from .data.functions import load_image
from .export import export_csv, export_html, export_markdown
from .layout_analyzer import LayoutAnalyzer
from .layout_parser import Element
from .ocr import OCR, WordPrediction
from .table_structure_recognizer import TableStructureRecognizerSchema
from .utils.misc import is_contained, quad_to_xyxy


class ParagraphSchema(BaseModel):
    box: conlist(int, min_length=4, max_length=4)
    contents: Union[str, None]
    direction: Union[str, None]


class DocumentAnalyzerSchema(BaseModel):
    paragraphs: List[ParagraphSchema]
    tables: List[TableStructureRecognizerSchema]
    words: List[WordPrediction]
    figures: List[Element]

    def to_html(self, out_path: str):
        export_html(self, out_path)

    def to_markdown(self, out_path: str):
        export_markdown(self, out_path)

    def to_csv(self, out_path: str):
        export_csv(self, out_path)

    def to_json(self, out_path: str):
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                self.dict(),
                f,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
                separators=(",", ": "),
            )


def combine_flags(flag1, flag2):
    return [f1 or f2 for f1, f2 in zip(flag1, flag2)]


def extract_words_within_element(pred_words, element):
    contained_words = []
    word_sum_width = 0
    word_sum_height = 0
    check_list = [False] * len(pred_words)
    for i, word in enumerate(pred_words):
        word_box = quad_to_xyxy(word.points)
        if is_contained(element.box, word_box, threshold=0.6):
            contained_words.append(word)
            word_sum_width += word_box[2] - word_box[0]
            word_sum_height += word_box[3] - word_box[1]
            check_list[i] = True

    if len(contained_words) == 0:
        return None, None, check_list

    mean_width = word_sum_width / len(contained_words)
    mean_height = word_sum_height / len(contained_words)

    word_direction = [word.direction for word in contained_words]
    cnt_horizontal = word_direction.count("horizontal")
    cnt_vertical = word_direction.count("vertical")

    element_direction = (
        "horizontal" if cnt_horizontal > cnt_vertical else "vertical"
    )
    if element_direction == "horizontal":
        contained_words = sorted(
            contained_words,
            key=lambda x: (
                x.points[0][1] // int(mean_height),
                x.points[0][0],
            ),
        )
    else:
        contained_words = sorted(
            contained_words,
            key=lambda x: (
                x.points[1][0] // int(mean_width),
                x.points[1][1],
            ),
            reverse=True,
        )

    contained_words = "\n".join(
        [content.content for content in contained_words]
    )
    return (contained_words, element_direction, check_list)


class DocumentAnalyzer:
    def __init__(self, configs=None, device="cuda", visualize=True):
        default_configs = {
            "ocr": {
                "text_detector": {
                    "device": device,
                    "visualize": visualize,
                },
                "text_recognizer": {
                    "device": device,
                    "visualize": visualize,
                },
            },
            "layout_analyzer": {
                "layout_parser": {
                    "device": device,
                    "visualize": visualize,
                },
                "table_structure_recognizer": {
                    "device": device,
                    "visualize": visualize,
                },
            },
        }

        if configs is None:
            configs = default_configs

        self.ocr = OCR(configs=configs["ocr"])
        self.layout = LayoutAnalyzer(configs=configs["layout_analyzer"])

    def aggregate(self, ocr_res, layout_res):
        paragraphs = []
        check_list = [False] * len(ocr_res.words)
        for table in layout_res.tables:
            for cell in table.cells:
                words, direction, flags = extract_words_within_element(
                    ocr_res.words, cell
                )

                if words is None:
                    words = ""

                cell.contents = words
                check_list = combine_flags(check_list, flags)

        for paragraph in layout_res.paragraphs:
            words, direction, flags = extract_words_within_element(
                ocr_res.words, paragraph
            )

            if words is None:
                continue

            paragraph = {
                "contents": words,
                "box": paragraph.box,
                "direction": direction,
            }
            check_list = combine_flags(check_list, flags)
            paragraph = ParagraphSchema(**paragraph)
            paragraphs.append(paragraph)

        for i, word in enumerate(ocr_res.words):
            direction = word.direction
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
            "tables": layout_res.tables,
            "figures": layout_res.figures,
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

            results_ocr, ocr = results[0]
            results_layout, layout = results[1]

        outputs = self.aggregate(results_ocr, results_layout)
        results = DocumentAnalyzerSchema(**outputs)
        return results, ocr, layout


if __name__ == "__main__":
    analyzer = DocumentAnalyzer()
    dir = "dataset/test"
    # dir = "dataset/test"
    outdir = "results"

    os.makedirs(outdir, exist_ok=True)

    for path_img in glob.glob(os.path.join(dir, "*.jpg")):
        img = load_image(path_img)
        results, ocr, layout = asyncio.run(analyzer(img))
        basename = os.path.basename(path_img)
        cv2.imwrite(
            os.path.join(outdir, f"{os.path.splitext(basename)[0]}_ocr.jpg"),
            ocr,
        )
        cv2.imwrite(
            os.path.join(
                outdir, f"{os.path.splitext(basename)[0]}_layout.jpg"
            ),
            layout,
        )
        html_path = os.path.join(
            outdir, f"{os.path.splitext(basename)[0]}.html"
        )
        results.to_html(html_path)

        markdown_path = os.path.join(
            outdir, f"{os.path.splitext(basename)[0]}.md"
        )
        results.to_markdown(markdown_path)

        csv_path = os.path.join(outdir, f"{os.path.splitext(basename)[0]}.csv")

        results.to_csv(csv_path)

        json_path = os.path.join(
            outdir, f"{os.path.splitext(basename)[0]}_result.json"
        )
        results.to_json(json_path)
