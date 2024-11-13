import asyncio
from concurrent.futures import ThreadPoolExecutor

from .data.functions import load_image
from .layout_alalyzer import LayoutAnalyzer
from .ocr import OCR
from .utils.export import export_html, export_md
from .utils.misc import is_contained


def quad_to_xyxy(quad):
    x1 = min([x for x, _ in quad])
    y1 = min([y for _, y in quad])
    x2 = max([x for x, _ in quad])
    y2 = max([y for _, y in quad])

    return x1, y1, x2, y2


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
        check_list = [False] * len(ocr_res["words"])
        for paragraph_box in layout_res["paragraph"]["boxes"]:
            words = []

            word_sum_width = 0
            word_sum_height = 0
            for i, word in enumerate(ocr_res["words"]):
                word_box = quad_to_xyxy(word["points"])
                if is_contained(paragraph_box, word_box):
                    words.append(word)
                    word_sum_width += word_box[2] - word_box[0]
                    word_sum_height += word_box[3] - word_box[1]
                    check_list[i] = True

            if len(words) == 0:
                continue

            mean_width = word_sum_width / len(words)
            mean_height = word_sum_height / len(words)

            if mean_width > mean_height:
                words = sorted(
                    words,
                    key=lambda x: (
                        x["points"][0][1] // int(mean_height),
                        x["points"][0][0],
                    ),
                )
            else:
                words = sorted(
                    words,
                    key=lambda x: (
                        x["points"][1][0] // int(mean_width),
                        x["points"][1][1],
                    ),
                    reverse=True,
                )

            words = "\n".join([content["content"] for content in words])
            paragraphs.append({"contents": words, "box": paragraph_box})

        for table in layout_res["table"]:
            for cell in table["cells"]:
                words = []

                word_sum_width = 0
                word_sum_height = 0
                for i, word in enumerate(ocr_res["words"]):
                    word_box = quad_to_xyxy(word["points"])
                    if is_contained(cell["box"], word_box, threshold=0.6):
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
                            x["points"][0][1] // int(mean_height),
                            x["points"][0][0],
                        ),
                    )
                else:
                    words = sorted(
                        words,
                        key=lambda x: (
                            x["points"][1][0] // int(mean_width),
                            x["points"][1][1],
                        ),
                        reverse=True,
                    )

                words = "\n".join([content["content"] for content in words])
                cell["contents"] = words
                cell["direction"] = direction

        for i, word in enumerate(ocr_res["words"]):
            direction = "horizontal"
            height = word["points"][1][1] - word["points"][0][1]
            width = word["points"][1][0] - word["points"][0][0]

            if 3 * height > width:
                direction = "vertical"

            if not check_list[i]:
                paragraphs.append(
                    {
                        "contents": word["content"],
                        "box": quad_to_xyxy(word["points"]),
                        "direction": direction,
                    }
                )

        ocr_res["paragraphs"] = paragraphs
        ocr_res["table"] = layout_res["table"]

        return ocr_res

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

        aggregate_res = self.aggregate(results_ocr, results_layout)
        return aggregate_res


if __name__ == "__main__":
    analyzer = DocumentAnalyzer()
    img = "dataset/test/00122864_3344358_13.jpg"
    img = load_image(img)
    results = asyncio.run(analyzer(img))
    export_html(results)
    export_md(results)
