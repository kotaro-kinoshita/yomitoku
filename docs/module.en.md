# Module Usage

This page explains how to use YomiToku as a Python library.

## Using Document Analyzer

The Document Analyzer performs OCR and layout analysis, integrating these results into a comprehensive analysis output. It can be used for various use cases, including paragraph and table structure analysis, extraction, and figure/table detection.

Following 4 models are utilized in the module:

- Text Recognizer
- Text Detector
- Layout Parser
- Table Structure Recognizer

<!--codeinclude-->
[demo/simple_document_analysis.py](../demo/simple_document_analysis.py)
<!--/codeinclude-->

| Option Name | Type | Description | Notes |
| :--- | :--- | :--- | :--- |
| `visualize` | `bool` | Specifies whether to visualize the processing results. | We recommend `False` if not for debugging. If `True`, the OCR results are returned as the 2nd return value and the layout analysis results as the 3rd return value. If `False`, `None` is returned. |
| `device` | `str` | Specifies the device to be used for processing. | The default is `"cuda"`. If a GPU is unavailable, it automatically switches to `"cpu"`. |
| `configs` | `dict` | Used to set more detailed parameters for module processing. | Refer to [Model Detailed Config](#model-config) for details. |
| `ignore_ruby` | `bool` | Specifies whether to exclude ruby (furigana) text from the output. | Default is `False`. |
| `ruby_threshold` | `float` | Specifies the threshold for ruby detection as a ratio to the median line height. Text consisting solely of hiragana or katakana below this threshold is identified as ruby. | Default is `0.5`. Only effective when `ignore_ruby=True`. |

The results of DocumentAnalyzer can be exported in the following formats:

| Method | Output Format |
| :--- | :--- |
| `to_json()` | JSON format (\*.json) |
| `to_html()` | HTML format (\*.html) |
| `to_csv()` | Comma-separated CSV format (\*.csv) |
| `to_markdown()` | Markdown format (\*.md) |

## Using AI-OCR Only

AI-OCR performs text detection and then executes recognition processing on the detected text to return the position (location) of the characters within the image and the reading result.

Following 2 models are utilized in the module:

- Text Recognizer
- Text Detector

<!--codeinclude-->
[demo/simple_ocr.py](../demo/simple_ocr.py)
<!--/codeinclude-->

| Option Name | Type | Description | Notes |
| :--- | :--- | :--- | :--- |
| `visualize` | `bool` | Specifies whether to visualize the processing results. | We recommend `False` if not for debugging. If `True`, the OCR results are returned as the 2nd return value. If `False`, `None` is returned. |
| `device` | `str` | Specifies the device to be used for processing. | The default is `"cuda"`. If a GPU is unavailable, it automatically switches to `"cpu"`. |
| `configs` | `dict` | Used to set more detailed parameters for module processing. | Refer to [Model Detailed Config](#model-config) for details. |

The results of OCR processing support export in JSON format (`to_json()`) only.

## Using Layout Analyzer only

The `LayoutAnalyzer` performs text detection, followed by AI-based paragraph, figure/table detection, and table structure analysis. It analyzes the layout structure within the document.

Following 2 models are utilized in the module:

- Layout Parser
- Table Structure Recognizer

<!--codeinclude-->
[demo/simple_layout.py](../demo/simple_layout.py)
<!--/codeinclude-->

| Option Name | Type | Description | Notes |
| :--- | :--- | :--- | :--- |
| `visualize` | `bool` | Specifies whether to visualize the processing results. | We recommend `False` if not for debugging. If `True`, the layout analysis results as the 2nd return value. If `False`, `None` is returned. |
| `device` | `str` | Specifies the device to be used for processing. | The default is `"cuda"`. If a GPU is unavailable, it automatically switches to `"cpu"`. |
| `configs` | `dict` | Used to set more detailed parameters for module processing. | Refer to [Model Detailed Config](#model-config) for details. |

The results of LayoutAnalyzer processing support export only in JSON format (to_json()).

## Model Detailed Config {:#model-config}

By providing a config, you can adjust the behavior in greater detail.
The following parameters can be set for the model:

| Option Name | Type | Description |
| :--- | :--- | :--- |
| `model_name` | `str` | Specifies the model name to be used. |
| `path_cfg` | `str` | Inputs the path to the config file containing the hyperparameters. |
| `device` | `str` | Specifies the device to be used for inference. (Allowed Values: `cuda` \| `cpu` \| `mps`) |
| `visualize` | `bool` | Specifies whether to perform visualization. |
| `from_pretrained` | `bool` | Specifies whether to use a Pretrained Model (a previously trained model). |
| `infer_onnx` | `bool` | Specifies whether to use ONNX Runtime instead of PyTorch for inference. |

### Supported Model Names (`model_name`)

| Model Type | Model Name |
| :--- | :--- |
| `TextRecognizer` | `"parseq"`, `"parseq-small"`, `"parseq-tiny"` |
| `TextDetector` | `"dbnet"` |
| `LayoutParser` | `"rtdetrv2"` |
| `TableStructureRecognizer` | `"rtdetrv2"` |

### How to Write Config

The config is provided in dictionary format. By using a config, you can execute processing on different devices for each model and set detailed parameters.

For example, the following config allows the OCR processing to run on a GPU, while the layout analysis is performed on a CPU:

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

## Defining Parameters in an YAML File

By providing the path to a YAML file in the config, you can adjust detailed parameters for inference. Examples of YAML files can be found in the `configs` directory within the repository.
While the model's network parameters cannot be modified, certain aspects like post-processing parameters and input image size can be adjusted. Refer to [Model Config](configuration.en.md) for configurable parameters.

For instance, you can define post-processing thresholds for the Text Detector in a YAML file and set its path in the config. The config file does not need to include all parameters; you only need to specify the parameters that require changes.

```text_detector.yaml
post_process:
  thresh: 0.1
  unclip_ratio: 2.5
```

The path to the YAML file can be stored in the Config, as follows:

<!--codeinclude-->
[demo/setting_document_anaysis.py](../demo/setting_document_anaysis.py)
<!--/codeinclude-->
