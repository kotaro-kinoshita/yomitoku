[日本語版](README.md) | English

<img src="static/logo/horizontal.png" width="800px">

![Python](https://img.shields.io/badge/Python-3.10|3.11|3.12-F9DC3E.svg?logo=python&logoColor=&style=flat)
![Pytorch](https://img.shields.io/badge/Pytorch-2.5-EE4C2C.svg?logo=Pytorch&style=fla)
![CUDA](https://img.shields.io/badge/CUDA->=11.8-76B900.svg?logo=NVIDIA&style=fla)
![OS](https://img.shields.io/badge/OS-Linux|Mac|Win-1793D1.svg?&style=fla)
[![Document](https://img.shields.io/badge/docs-live-brightgreen)](https://kotaro-kinoshita.github.io/yomitoku-dev/)

## 🌟 Introduction

YomiToku is a Document AI engine specialized in Japanese document image analysis. It provides full OCR (optical character recognition) and layout analysis capabilities, enabling the recognition, extraction, and conversion of text and diagrams from images.

- 🤖 Equipped with four AI models trained on Japanese datasets: text detection, text recognition, layout analysis, and table structure recognition. All models are independently trained and optimized for Japanese documents, delivering high-precision inference.
- 🇯🇵 Each model is specifically trained for Japanese document images, supporting the recognition of over 7,000 Japanese characters, including vertical text and other layout structures unique to Japanese documents. (It also supports English documents.)
- 📈 By leveraging layout analysis, table structure parsing, and reading order estimation, it extracts information while preserving the semantic structure of the document layout.
- 📄 Supports a variety of output formats, including HTML, Markdown, JSON, and CSV. It also allows for the extraction of diagrams and images contained within the documents.
- ⚡ Operates efficiently in GPU environments, enabling fast document transcription and analysis. It requires less than 8GB of VRAM, eliminating the need for high-end GPUs.

## 🖼️ Demo

The verification results for various types of images are also included in [gallery.md](gallery.md)

|                           Input                            |                     Results of OCR                      |
| :--------------------------------------------------------: | :-----------------------------------------------------: |
|        <img src="static/in/demo.jpg" width="400px">        | <img src="static/out/in_demo_p1_ocr.jpg" width="400px"> |
|                 Results of Layout Analysis                 |                 Results of HTML Export                  |
| <img src="static/out/in_demo_p1_layout.jpg" width="400px"> |   <img src="static/out/demo_html.png" width="400px">    |

For the results exported in Markdown, please refer to [static/out/in_demo_p1.md](static/out/in_demo_p1.md) in the repository.

- `Red Frame`: Positions of figures and images
- `Green Frame`: Overall table region
- `Pink Frame`:` Table cell structure (text within the cells represents [row number, column number] (rowspan x colspan))
- `Blue Frame`: Paragraph and text group regions
- `Red Arrow`: Results of reading order estimation

Source of the image: Created by processing content from “Reiwa 6 Edition Information and Communications White Paper, Chapter 3, Section 2: Technologies Advancing with AI Evolution” (https://www.soumu.go.jp/johotsusintokei/whitepaper/ja/r06/pdf/n1410000.pdf)：(Ministry of Internal Affairs and Communications).

## 📣 Release

- Released YomiToku v0.5.1 (beta) on November 26, 2024.

## 💡 Installation

```
pip install yomitoku
```

- Please install the version of PyTorch that matches your CUDA version. By default, a version compatible with CUDA 12.4 or higher will be installed.
- PyTorch versions 2.5 and above are supported. As a result, CUDA version 11.8 or higher is required. If this is not feasible, please use the Dockerfile provided in the repository.

## 🚀 Usage

```
yomitoku ${path_data} -f md -o results -v --figure --lite
```

- `${path_data}`: Specify the path to a directory containing images to be analyzed or directly provide the path to an image file. If a directory is specified, images in its subdirectories will also be processed.
- `-f`, `--format`: Specify the output file format. Supported formats are json, csv, html, and md.
- `-o`, `--outdir`: Specify the name of the output directory. If it does not exist, it will be created.
- `-v`, `--vis`: If specified, outputs visualized images of the analysis results.
- `-l`, `--lite`: inference is performed using a lightweight model. This enables fast inference even on a CPU.
- `-d`, `--device`: Specify the device for running the model. If a GPU is unavailable, inference will be executed on the CPU. (Default: cuda)
- `--ignore_line_break`: Ignores line breaks in the image and concatenates sentences within a paragraph. (Default: respects line breaks as they appear in the image.)
- `--figure_letter`: Exports characters contained within detected figures and tables to the output file.
- `--figure`: Exports detected figures and images to the output file
- `--encoding` Specifies the character encoding for the output file to be exported. If unsupported characters are included, they will be ignored. (utf-8, utf-8-sig, shift-jis, enc-jp, cp932)
- `--combine` When a PDF is provided as input and contains multiple pages, this option combines their prediction results into a single file for export.
- `--ignore_meta` Excludes text information such as headers and footers from the output file.

For other options, please refer to the help documentation.

```
yomitoku --help
```

**NOTE**

- It is recommended to run on a GPU. The system is not optimized for inference on CPUs, which may result in significantly longer processing times.
- Only printed text recognition is supported. While it may occasionally read handwritten text, official support is not provided.
- YomiToku is optimized for document OCR and is not designed for scene OCR (e.g., text printed on non-paper surfaces like signs).
- The resolution of input images is critical for improving the accuracy of AI-OCR recognition. Low-resolution images may lead to reduced recognition accuracy. It is recommended to use images with a minimum short side resolution of 720px for inference.

## 📝 Documents

For more details, please refer to the [documentation](https://kotaro-kinoshita.github.io/yomitoku-dev/)

## LICENSE

The source code stored in this repository and the model weight files related to this project on Hugging Face Hub are licensed under CC BY-NC-SA 4.0.
You are free to use them for non-commercial personal use or research purposes.
For commercial use, a separate commercial license is available. Please contact the developers for more information.

YomiToku © 2024 by Kotaro Kinoshita is licensed under CC BY-NC-SA 4.0. To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
