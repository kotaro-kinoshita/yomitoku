# Usage

## CLI

The model weight files are downloaded from Hugging Face Hub only during the first execution.

```
yomitoku ${path_data} -f md -o results -v
```

- `${path_data}`: Specify the path to a directory containing images to be analyzed or directly provide the path to an image file. If a directory is specified, images in its subdirectories will also be processed.
- `-f`, `--format`: Specify the output file format. Supported formats are json, csv, html, and md.
- `-o`, `--outdir`: Specify the name of the output directory. If it does not exist, it will be created.
- `-v`, `--vis`: If specified, outputs visualized images of the analysis results.
- `-l`, `--lite`: inference is performed using a lightweight model. This enables fast inference even on a CPU.
- `-d`, `--device`: Specify the device for running the model. If a GPU is unavailable, inference will be executed on the CPU. (Default: cuda)
- `--ignore_line_break`: Ignores line breaks in the image and concatenates sentences within a paragraph. (Default: respects line breaks as they appear in the image.)
- `--figure_letter`: Exports characters contained within detected figures and tables to the output file.
- `--figure`: Exports detected figures and images to the output file (supported only for html and markdown).
- `--encoding` Specifies the character encoding for the output file to be exported. If unsupported characters are included, they will be ignored. (utf-8, utf-8-sig, shift-jis, enc-jp, cp932)


**NOTE**

- It is recommended to run on a GPU. The system is not optimized for inference on CPUs, which may result in significantly longer processing times.
- Only printed text recognition is supported. While it may occasionally read handwritten text, official support is not provided.
- YomiToku is optimized for document OCR and is not designed for scene OCR (e.g., text printed on non-paper surfaces like signs).
- The resolution of input images is critical for improving the accuracy of AI-OCR recognition. Low-resolution images may lead to reduced recognition accuracy. It is recommended to use images with a minimum short side resolution of 720px for inference.

## Calling from within Python code

### Document Analyzer の利用

The Document Analyzer performs OCR and layout analysis, integrating these results into a comprehensive analysis output. It can be used for various use cases, including paragraph and table structure analysis, extraction, and figure/table detection.

<!--codeinclude-->

[demo/simple_document_analysis.py](../demo/simple_document_analysis.py)

<!--/codeinclude-->

- Setting `visualize` to True enables the visualization of each processing result. The second and third return values will contain the OCR and layout analysis results, respectively. If set to False, None will be returned. Since visualization adds computational overhead, it is recommended to set it to False unless needed for debugging purposes.
- The `device` parameter specifies the computation device to be used. The default is "cuda". If a GPU is unavailable, it automatically switches to CPU mode for processing.
- The `configs` parameter allows you to set more detailed parameters for the pipeline processing.

The results of DocumentAnalyzer can be exported in the following formats:

`to_json()`: JSON format (_.json)
`to_html()`: HTML format (_.html)
`to_csv()`: Comma-separated CSV format (_.csv)
`to_markdown()`: Markdown format (_.md)

### Using AI-OCR Only

AI-OCR performs text detection and recognition on the detected text, returning the positions of the text within the image along with the

<!--codeinclude-->

[demo/simple_ocr.py](../demo/simple_ocr.py)

<!--/codeinclude-->

- Setting `visualize` to True enables the visualization of each processing result. The second and third return values will contain the OCR and layout analysis results, respectively. If set to False, None will be returned. Since visualization adds computational overhead, it is recommended to set it to False unless needed for debugging purposes.
- The `device` parameter specifies the computation device to be used. The default is "cuda". If a GPU is unavailable, it automatically switches to CPU mode for processing.
- The `configs` parameter allows you to set more detailed parameters for the pipeline processing.

The results of OCR processing support export in JSON format (`to_json()`) only.

### Using Layout Analyzer only

The `LayoutAnalyzer` performs text detection, followed by AI-based paragraph, figure/table detection, and table structure analysis. It analyzes the layout structure within the document.

<!--codeinclude-->

[demo/simple_layout.py](../demo/simple_layout.py)

<!--/codeinclude-->

- Setting `visualize` to True enables the visualization of each processing result. The second and third return values will contain the OCR and layout analysis results, respectively. If set to False, None will be returned. Since visualization adds computational overhead, it is recommended to set it to False unless needed for debugging purposes.
- The `device` parameter specifies the computation device to be used. The default is "cuda". If a GPU is unavailable, it automatically switches to CPU mode for processing.
- The `configs` parameter allows you to set more detailed parameters for the pipeline processing.

The results of LayoutAnalyzer processing support export only in JSON format (to_json()).

## Detailed Configuration of the Pipeline

By providing a config, you can adjust the behavior in greater detail.

### How to Write a Config

The config is provided in dictionary format. By using a config, you can execute processing on different devices for each module and set detailed parameters. For example, the following config allows the OCR processing to run on a GPU, while the layout analysis is performed on a CPU:

```python
from yomitoku import DocumentAnalyzer

if __name__ == "__main__":
    configs = {
        "ocr": {
            "text_detector": {
                "device": "cuda",
            },
            "text_recognizer": {
                "device": "cuda",
            },
        },
        "layout_analyzer": {
            "layout_parser": {
                "device": "cpu",
            },
            "table_structure_recognizer": {
                "device": "cpu",
            },
        },
    }

    DocumentAnalyzer(configs=configs)
```

### Defining Parameters in an YAML File

By providing the path to a YAML file in the config, you can adjust detailed parameters for inference. Examples of YAML files can be found in the `configs` directory within the repository. While the model's network parameters cannot be modified, certain aspects like post-processing parameters and input image size can be adjusted.

For instance, you can define post-processing thresholds for the Text Detector in a YAML file and set its path in the config. The config file does not need to include all parameters; you only need to specify the parameters that require changes.

```text_detector.yaml
post_process:
  thresh: 0.1
  unclip_ratio: 2.5
```

Storing the Path to a YAML File in the Config

<!--codeinclude-->

[demo/setting_document_anaysis.py](../demo/setting_document_anaysis.py)

<!--/codeinclude-->

## Using in an Offline Environment

Yomitoku automatically downloads models from Hugging Face Hub during the first execution, requiring an internet connection at that time. However, by manually downloading the models in advance, it can be executed in an offline environment.

1. Install [Git Large File Storage](https://docs.github.com/ja/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)
2. In an environment with internet access, download the model repository. Copy the cloned repository to your target environment using your preferred tools.

The following is the command to download the model repository from Hugging Face Hub.

```sh
git clone https://huggingface.co/KotaroKinoshita/yomitoku-table-structure-recognizer-rtdtrv2-open-beta

git clone https://huggingface.co/KotaroKinoshita/yomitoku-layout-parser-rtdtrv2-open-beta

git clone https://huggingface.co/KotaroKinoshita/yomitoku-text-detector-dbnet-open-beta

git clone https://huggingface.co/KotaroKinoshita/yomitoku-text-recognizer-parseq-open-beta
```

3. Place the model repository directly under the root directory of the Yomitoku repository and reference the local model repository in the `hf_hub_repo` field of the YAML file. Below is an example of `text_detector.yaml`. Similarly, define YAML files for other modules as well.

```yaml
hf_hub_repo: yomitoku-text-detector-dbnet-open-beta
```

4. Storing the Path to a YAML File in the Config

<!--codeinclude-->

[demo/setting_document_anaysis.py](../demo/setting_document_anaysis.py)

<!--/codeinclude-->
