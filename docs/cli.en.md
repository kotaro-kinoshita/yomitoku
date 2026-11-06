# CLI Usage

This page explains how to use YomiToku as a command-line interface (CLI).

The model weight files are downloaded from Hugging Face Hub only during the first execution.

```bash
yomitoku ${path_data} -v -o results
```

| Option Name | Description |
| :-- | :-- |
| `${path_data}` | Specifies the path to the directory containing images or the path to an image file. |
| `-o`, `--outdir` | Specifies the output directory (will be created if it doesn't exist). |
| `-v`, `--vis` | Outputs a visualization image of the analysis results. |

**Supplement: About `${path_data}`**

- An image file or a directory can be specified.
- If a directory is specified, it will be processed recursively, including subdirectories.
- The supported file formats are `pdf`, `jpeg`, `png`, `bmp`, and `tiff`.

!!! note
    - Only printed text recognition is supported. While it may occasionally read handwritten text, official support is not provided.
    - YomiToku is optimized for document OCR and is not designed for scene OCR (e.g., text printed on non-paper surfaces like signs).
    - The resolution of input images is critical for improving the accuracy of AI-OCR recognition. Low-resolution images may lead to reduced recognition accuracy. It is recommended to use images with a minimum short side resolution of 720px for inference.

## Reference for Help

Displays the options available for the CLI using 　`--help`, `-h`

```bash
yomitoku -h
```

## Running in Lightweight Mode

By using the --lite option, it is possible to perform inference with a lightweight model. This enables faster analysis compared to the standard mode. However, the accuracy of character recognition may decrease.

```bash
yomitoku ${path_data} --lite -v
```

In lightweight mode, the text recognizer is switched to the compact
`parseq-tiny-dynw-v4` model (32px input height, 192-dim, `[4, 8]` patch),
which is trained for **dynamic-width** inference. Instead of padding every
text crop to the fixed 800px canvas, each mini-batch is padded only to the
widest crop it contains, and similar-width crops are grouped together
(**batch bucketing**). Because most text crops are narrow, this removes a
large amount of wasted computation and is dramatically faster on CPU. On CPU,
the text detector additionally runs with the ONNX runtime and recognizer
mini-batches are executed concurrently.

The lightweight recognizer is downloaded from Hugging Face Hub
(`KotaroKinoshita/yomitoku-text-recognizer-parseq-tiny-dynw-v4`) on first use.

## Specifying Output Format

You can specify the output format of the analysis results using the --format or -f option. Supported output formats include JSON, CSV, HTML, and MD (Markdown).

```bash
yomitoku ${path_data} -f md
```

If `pdf` is specified, the system will recognize the text within the image using OCR and embed the text information as an invisible layer to convert it into a searchable PDF.

## Specifying the Output Device

You can specify the device for running the model using the -d or --device option. Supported options are cuda, cpu, and mps. If a GPU is not available, inference will be performed on the CPU. (Default: cuda)

```bash
yomitoku ${path_data} -d cpu
```

## Ignoring Line Breaks

In the normal mode, line breaks are applied based on the information described in the image. By using the --ignore_line_break option, you can ignore the line break positions in the image and return the same sentence within a paragraph as a single connected output.

```bash
yomitoku ${path_data} --ignore_line_break
```

## Outputting Figures and Graph Images

In the normal mode, information about figures or images contained in document images is not output. By using the --figure option, you can extract figures and images included in the document image, save them as separate image files, and include links to the detected individual images in the output file.

```bash
yomitoku ${path_data} --figure
```

## Outputting Text Contained in Figures and Images

In normal mode, text information contained within figures or images is not included in the output file. By using the --figure_letter option, text information within figures and images will also be included in the output file.

```bash
yomitoku ${path_data} --figure_letter
```

## Specifying the Character Encoding of the Output File

You can specify the character encoding of the output file using the --encoding option. Supported encodings include `utf-8`, `utf-8-sig`, `shift-jis`, `enc-jp`, and `cp932`. If unsupported characters are encountered, they will be ignored and not included in the output.

```bash
yomitoku ${path_data} --encoding utf-8-sig
```

## Specifying the Path to Config Files

The following options are used to specify the path to the YAML configuration file for each respective module.

| Option Name | Target Model |
| :--- | :--- |
| `--td_cfg` | Text Detector (TD) |
| `--tr_cfg` | Text Recognizer (TR) |
| `--lp_cfg` | Layout Parser (LP) |
| `--tsr_cfg` | Table Structure Recognizer (TSR) |

```bash
yomitoku ${path_data} --td_cfg ${path_yaml}
```

## Do not include metadata in the output file

You can exclude metadata such as headers and footers from the output file.

```bash
yomitoku ${path_data} --ignore_meta
```

## Split text spanning table cells

Text lines detected across multiple table cells are split at the cell boundaries before recognition, so that each cell receives only its own text.

```bash
yomitoku ${path_data} --split_text_across_cells
```

## Combine multiple pages

If the PDF contains multiple pages, you can export them as a single file.

```bash
yomitoku ${path_data} -f md --combine
```

## PDF Output Image Quality

You can specify the image quality preset for searchable PDF output using `--pdf_quality`. The default is `high`.

| Preset | Max Long Side | JPEG Quality | Description |
| :--- | :--- | :--- | :--- |
| `high` | No limit | 85 | High quality (default). Preserves the original image resolution. |
| `middle` | 2000px | 80 | Medium quality. Balances file size and image quality. |
| `low` | 1500px | 60 | Low quality. Minimizes file size. |

```bash
yomitoku ${path_data} -f pdf --pdf_quality middle
```

## Setting the PDF Reading Resolution

Specifies the resolution (DPI) when reading a PDF (default DPI = 200). Increasing the DPI value may improve recognition accuracy when dealing with fine text or small details within the PDF.

```bash
yomitoku ${path_data} --dpi 250
```

## Specifying Reading Order

By default, the reading order option is set to `auto`.

When `auto` is specified, the system identifies the document's orientation (horizontal or vertical) and automatically estimates the reading order. Specifically, the order is estimated as `top2left` for horizontal documents and `top2bottom` for vertical documents.

The reading order can also be specified manually, as follows:

```bash
yomitoku ${path_data} --reading_order left2right
```

| Setting Name | Preferred Reading Order | Valid Document Types |
| :--- | :--- | :--- |
| `top2bottom` | Top to Bottom | Column-formatted Word documents, etc. |
| `left2right` | Left to Right | Layouts where keys and values are in columns (e.g., receipts, insurance cards) |
| `right2left` | Right to Left | Vertically written documents |

## Enabling Recognition Orientation Fallback

By default, orientation fallback is disabled. When enabled with `--enable-rec-orientation-fallback`, if the confidence score of text recognition is low, the system retries recognition with the ROI image rotated 180 degrees and adopts the result with the higher confidence.

```bash
yomitoku ${path_data} --enable-rec-orientation-fallback
```

You can specify the confidence threshold for triggering the fallback using `--rec-orientation-fallback-thresh`. (Default: 0.75)

```bash
yomitoku ${path_data} --enable-rec-orientation-fallback --rec-orientation-fallback-thresh 0.6
```

## Disabling Repetition Stop

During text recognition, when a loop of abnormally repeated characters or strings is detected, decoding stops at that point and the repetition is collapsed to a single unit. This is enabled by default. If you want to keep long repetitions as they are, such as leader dots (`……`) or long runs of symbols, disable it with `--disable-repetition-stop`.

```bash
yomitoku ${path_data} --disable-repetition-stop
```

The detection thresholds can be tuned in the YAML passed to `--tr_cfg`. See [Model Configuration](configuration.md) for details.

## Excluding Ruby (Furigana) Text

You can exclude ruby (furigana) text from the output. When the `--ignore_ruby` option is set, text whose line height is below a certain threshold relative to the median line height within each paragraph or cell, and consists solely of hiragana or katakana characters, is identified as ruby and excluded.

```bash
yomitoku ${path_data} --ignore_ruby
```

You can adjust the ruby detection threshold using the `--ruby_threshold` option (default: 0.5). Increasing the value widens the range of text identified as ruby.

```bash
yomitoku ${path_data} --ignore_ruby --ruby_threshold 0.6
```

## Specifying Pages to Process

You can choose to process only specific pages.
Pages can be specified either as a comma-separated list or as a range using a hyphen.

```bash
yomitoku ${path_data} --pages 1,3-5,10
```

## Geometric Correction (Dewarping)

With the `--dewarp` option, the curvature, curl, tilt and keystone distortion of documents photographed with a smartphone or similar are corrected before the analysis. Text lines and layout become straight in the dewarped image, which makes OCR and layout analysis more stable.

Rather than a simple correction that extracts the paper outline and applies a projective transform to its four corners, it estimates the distortion over the whole page (non-linear correction), so curved and curled paper is flattened as well. It is robust to photos where part of the paper is cut off by the frame and to shadows and uneven lighting.

The default model (`pagenet_v1`) finds every page in the image (both pages of a spread, business cards or receipts side by side, ...) and dewarps and analyzes each page on its own. A spread is split at the binding into its left and right pages.

```bash
yomitoku ${path_data} -v --dewarp
```

| Option Name | Description |
| :--- | :--- |
| `--dewarp` | Enables the geometric correction. |
| `--dewarp_coordinate` | Coordinates of the output: `dewarped` (the dewarped image, default) or `photo` (the input image). |
| `--dewarp_model` | Dewarping model: `pagenet_v1` (finds and dewarps every page of the image, default) or `uvdoc_v1` (dewarps the image as one page). |
| `--dewarp_cfg` | Path to the config (YAML) of the dewarping model. |

Output files:

| File | Description |
| :--- | :--- |
| `*_dewarped.jpg` | The dewarped image (only when the correction succeeded). Results in `dewarped` coordinates refer to this image. |
| `*_dewarp_grid.jpg` | Only with `--vis`. `pagenet_v1`: for each page found, the grid of the whole page (thin lines), the part dewarped (thick line), the page index and the detection score. `uvdoc_v1`: the estimated paper region (yellow line, the outside darkened), where the rows (red) and columns (blue) of the dewarped image lie in the input image (grid), and the outer frame of the dewarped image (green). |

When `pagenet_v1` finds several pages in one image, the output files of each page are `*_p{page}_{k}.*` (`k` from 1, left to right then top to bottom); with a single page they stay `*_p{page}.*`. One `*_dewarp_grid.jpg` is written per image.

With `--dewarp_coordinate photo`, the results obtained on the dewarped image are mapped back to the input image. Every element (paragraph, table, cell, figure, word) gets a curved `polygon` on the input image and `box` becomes its bounding box. The visualizations are drawn on the input image as well.

```bash
yomitoku ${path_data} -f json -v --dewarp --dewarp_coordinate photo
```

### Detecting Failed Corrections

When the estimated correction is judged broken (the paper region and the grid disagree, the grid folds over itself, ...; with `pagenet_v1`, when no page is found), the page is analyzed on the input image without correction and a warning is logged. With `pagenet_v1`, pages whose grid collapsed, pages lying inside a larger page and pages mostly cut off by the frame (such as the neighbouring page at the edge of a spread) are dropped before the correction. The outcome is recorded in `preprocess.dewarp` of the JSON output.

```json
"preprocess": {
  "dewarp": {
    "status": "ok",
    "reasons": [],
    "coordinate": "dewarped",
    "size": [6792, 4120],
    "mask_iou": 0.9995,
    "flipped_cells": 0.022,
    "spacing_ratio": 2.34,
    "outside": 0.082
  }
}
```

| Field | Description |
| :--- | :--- |
| `status` | `ok`: corrected, `rejected`: judged failed and analyzed without correction |
| `reasons` | Reasons of `rejected` (`uvdoc_v1`: `mask_iou`, `flipped_cells`, `spacing`, `outside`; `pagenet_v1`: `no_page`) |
| `coordinate` | Coordinate system of the results (`dewarped` / `photo`); always `photo` when `rejected` |
| `size` | Size of the dewarped image `[width, height]` |
| `mask_iou` | IoU of the estimated paper region and the grid outline (`uvdoc_v1` only) |
| `flipped_cells` | Share of grid cells turned over |
| `spacing_ratio` | Longest grid edge / median grid edge length |
| `outside` | Share of grid points far outside the image |
| `page` | `pagenet_v1` only. Index of the page among the pages found in the image (from 0, left to right then top to bottom) |
| `num_pages` | `pagenet_v1` only. Number of pages found in the image |
| `score` | `pagenet_v1` only. Page detection score |

!!! note
    - The geometric correction targets documents photographed with a smartphone or similar. Do not use it for documents that are already flat, such as scans or PDFs: without the paper outline the correction brings nothing and may slightly distort the page.
    - The correction of creases is limited.
    - In the dewarped image, what lies outside the paper (desk, outside the photo) is painted with the paper colour.
