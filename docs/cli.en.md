# CLI Usage

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

* An image file or a directory can be specified.
* If a directory is specified, it will be processed recursively, including subdirectories.
* The supported file formats are `pdf`, `jpeg`, `png`, `bmp`, and `tiff`.

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

## Specifying Output Format

You can specify the output format of the analysis results using the --format or -f option. Supported output formats include JSON, CSV, HTML, and MD (Markdown).

```bash
yomitoku ${path_data} -f md
```

If a PDF file is specified, the system will recognize the text within the image using OCR and embed the text information as an invisible layer to convert it into a searchable PDF.

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

| Option Name | Target Module |
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

## Combine multiple pages

If the PDF contains multiple pages, you can export them as a single file.

```bash
yomitoku ${path_data} -f md --combine
```

## Setting the PDF Reading Resolution

Specifies the resolution (DPI) when reading a PDF (default DPI = 200). Increasing the DPI value may improve recognition accuracy when dealing with fine text or small details within the PDF.

```bash
yomitoku ${path_data} --dpi 250
```

## Specifying Reading Order

By default, the reading order option is set to `auto`.

When `auto` is specified, the system identifies the document's orientation (horizontal or vertical) and automatically estimates the reading order. Specifically, the order is estimated as `top2left` for horizontal documents and `top2bottom` for vertical documents.

"The reading order can also be specified manually, as follows:

```bash
yomitoku ${path_data} --reading_order left2right
```

| Setting Name | Preferred Reading Order | Valid Document Types |
| :--- | :--- | :--- |
| `top2bottom` | Top to Bottom | Column-formatted Word documents, etc. |
| `left2right` | Left to Right | Layouts where keys and values are in columns (e.g., receipts, insurance cards) |
| `right2left` | Right to Left | Vertically written documents |

## Specifying Pages to Process

You can choose to process only specific pages.
Pages can be specified either as a comma-separated list or as a range using a hyphen.

```bash
yomitoku ${path_data} --pages 1,3-5,10
```
