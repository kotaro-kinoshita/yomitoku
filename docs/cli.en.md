# CLI Usage

The model weight files are downloaded from Hugging Face Hub only during the first execution.

```
yomitoku ${path_data} -v -o results
```

- `${path_data}`: Specify the path to a directory containing images to be analyzed or directly provide the path to an image file. If a directory is specified, images in its subdirectories will also be processed.
- `-f`, `--format`: Specify the output file format. Supported formats are json, csv, html, and md.
- `-o`, `--outdir`: Specify the name of the output directory. If it does not exist, it will be created.
- `-v`, `--vis`: If specified, outputs visualized images of the analysis results.

**NOTE**

- Only printed text recognition is supported. While it may occasionally read handwritten text, official support is not provided.
- YomiToku is optimized for document OCR and is not designed for scene OCR (e.g., text printed on non-paper surfaces like signs).
- The resolution of input images is critical for improving the accuracy of AI-OCR recognition. Low-resolution images may lead to reduced recognition accuracy. It is recommended to use images with a minimum short side resolution of 720px for inference.

## Reference for Help

Displays the options available for the CLI using 　`--help`, `-h`

```
yomitoku -h
```

## Running in Lightweight Mode

By using the --lite option, it is possible to perform inference with a lightweight model. This enables faster analysis compared to the standard mode. However, the accuracy of character recognition may decrease.

```
yomitoku ${path_data} --lite -v
```

## Specifying Output Format

You can specify the output format of the analysis results using the --format or -f option. Supported output formats include JSON, CSV, HTML, and MD (Markdown).

```
yomitoku ${path_data} -f md
```

## Specifying the Output Device

You can specify the device for running the model using the -d or --device option. Supported options are cuda, cpu, and mps. If a GPU is not available, inference will be performed on the CPU. (Default: cuda)

```
yomitoku ${path_data} -d cpu
```

## Ignoring Line Breaks

In the normal mode, line breaks are applied based on the information described in the image. By using the --ignore_line_break option, you can ignore the line break positions in the image and return the same sentence within a paragraph as a single connected output.

```
yomitoku ${path_data} --ignore_line_break
```

## Outputting Figures and Graph Images

In the normal mode, information about figures or images contained in document images is not output. By using the --figure option, you can extract figures and images included in the document image, save them as separate image files, and include links to the detected individual images in the output file.

```
yomitoku ${path_data} --figure
```

## Outputting Text Contained in Figures and Images

In normal mode, text information contained within figures or images is not included in the output file. By using the --figure_letter option, text information within figures and images will also be included in the output file.

```
yomitoku ${path_data} --figure_letter
```

## Specifying the Character Encoding of the Output File

You can specify the character encoding of the output file using the --encoding option. Supported encodings include `utf-8`, `utf-8-sig`, `shift-jis`, `enc-jp`, and `cp932`. If unsupported characters are encountered, they will be ignored and not included in the output.

```
yomitoku ${path_data} --encoding utf-8-sig
```

## Specifying the Path to Config Files

Specify the path to the config files for each module as follows:

- `--td_cfg`: Path to the YAML file containing the config for the Text Detector
- `--tr_cfg`: Path to the YAML file containing the config for the Text Recognizer
- `--lp_cfg`: Path to the YAML file containing the config for the Layout Parser
- `--tsr_cfg`: Path to the YAML file containing the config for the Table Structure Recognizer

```
yomitoku ${path_data} --td_cfg ${path_yaml}
```

## Do not include metadata in the output file

You can exclude metadata such as headers and footers from the output file.
```
yomitoku ${path_data} --ignore_meta
```

## Combine multiple pages

If the PDF contains multiple pages, you can export them as a single file.

```
yomitoku ${path_data} -f md --combine
```

## Specifying Reading Order

By default, *Auto* mode automatically detects whether a document is written horizontally or vertically and estimates the appropriate reading order. However, you can explicitly specify a custom reading order. For horizontal documents, the default is `top2left`, and for vertical documents, it is `top2bottom`.

```
yomitoku ${path_data} --reading_order left2right
```

* `top2bottom`: Prioritizes reading from top to bottom. Useful for multi-column documents such as word processor files with vertical flow.

* `left2right`: Prioritizes reading from left to right. Suitable for layouts like receipts or health insurance cards, where key-value text pairs are arranged in columns.

* `right2left`: Prioritizes reading from right to left. Effective for vertically written documents.

## Create a searchable PDF

Detect the text in the image and embed it into the PDF as invisible text, converting the file into a searchable PDF.

```
yomitoku ${path_data} --searchable_pdf
```