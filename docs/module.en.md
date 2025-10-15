# Calling from within Python code

## Document Analyzer の利用

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

## Using AI-OCR Only

AI-OCR performs text detection and recognition on the detected text, returning the positions of the text within the image along with the

<!--codeinclude-->

[demo/simple_ocr.py](../demo/simple_ocr.py)

<!--/codeinclude-->

- Setting `visualize` to True enables the visualization of each processing result. The second and third return values will contain the OCR and layout analysis results, respectively. If set to False, None will be returned. Since visualization adds computational overhead, it is recommended to set it to False unless needed for debugging purposes.
- The `device` parameter specifies the computation device to be used. The default is "cuda". If a GPU is unavailable, it automatically switches to CPU mode for processing.
- The `configs` parameter allows you to set more detailed parameters for the pipeline processing.

The results of OCR processing support export in JSON format (`to_json()`) only.

## Using Layout Analyzer only

The `LayoutAnalyzer` performs text detection, followed by AI-based paragraph, figure/table detection, and table structure analysis. It analyzes the layout structure within the document.

<!--codeinclude-->

[demo/simple_layout.py](../demo/simple_layout.py)

<!--/codeinclude-->

- Setting `visualize` to True enables the visualization of each processing result. The second and third return values will contain the OCR and layout analysis results, respectively. If set to False, None will be returned. Since visualization adds computational overhead, it is recommended to set it to False unless needed for debugging purposes.
- The `device` parameter specifies the computation device to be used. The default is `cuda`. If a GPU is unavailable, it automatically switches to CPU mode for processing.
- The `configs` parameter allows you to set more detailed parameters for the pipeline processing.

The results of LayoutAnalyzer processing support export only in JSON format (to_json()).

## Detailed Configuration of the Pipeline

By providing a config, you can adjust the behavior in greater detail.

- model_name: Specifies the architecture of the model to be used.
- path_cfg: Provides the path to the config file containing hyperparameters.
- device: Specifies the device to be used for inference. Options are `cuda`, `cpu`, or `mps`.
- visualize: Indicates whether to perform visualization of the processing results (boolean).
- from_pretrained: Specifies whether to use a pretrained model (boolean).
- infer_onnx: Indicates whether to use onnxruntime for inference instead of PyTorch (boolean).

**Supported Model Types (model_name)**

- TextRecognizer: `parseq`, `parseq-small`
- TextDetector: `dbnet`
- LayoutParser: `rtdetrv2`
- TableStructureRecognizer: `rtdetrv2`

### How to Write Config

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

## Defining Parameters in an YAML File

By providing the path to a YAML file in the config, you can adjust detailed parameters for inference. Examples of YAML files can be found in the `configs` directory within the repository. While the model's network parameters cannot be modified, certain aspects like post-processing parameters and input image size can be adjusted.Refer to [configuration](configuration.en.mdmd) for configurable parameters.

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


## Using Yomitoku in Offline Environments

Yomitoku automatically downloads the model from Hugging Face Hub on its first run.
An internet connection is required at that time, but by manually downloading the model beforehand, you can also run Yomitoku in environments without internet access.

```
download_model
```

By placing the downloaded repository folder `KotaroKinoshita` in the current directory at runtime, the local repository model will be loaded and executed without any internet connection.

