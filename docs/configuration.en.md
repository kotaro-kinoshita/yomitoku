# Model Config

This page explains the configurable parameters for each model.

## How to Configure {: #how-to-configure}

The settings of each module (Text Detector / Text Recognizer / Layout Parser / Table Structure Recognizer) are determined by the following three layers. Higher layers take precedence.

| Priority | Layer | How to specify |
| :---: | :--- | :--- |
| High | Arguments | CLI options, constructor arguments in the Python API, and values in the `configs` dictionary |
| Middle | YAML | `--td_cfg` / `--tr_cfg` / `--lp_cfg` / `--tsr_cfg` in the CLI, `path_cfg` in the Python API |
| Low | Model defaults | Values defined in the config of the model selected with `model_name` |

- In YAML, you only need to write the items you want to change. Items you omit keep the model defaults. For nested items (e.g. `data.batch_size`), you can also write only the items you change.
- Only items that are available as constructor arguments of a module can be passed as arguments (e.g. `repetition_stop` and `rec_orientation_fallback` of the Text Recognizer). Other items, such as thresholds, are set in YAML. See the [Module Usage](module.en.md) for the available arguments.
- If the same item is set both in YAML and as an argument, the argument value is used.

### Where to Specify Settings

The keys of the `configs` dictionary depend on how you use YomiToku. Under each key, write the module's arguments (including `path_cfg`) as a dictionary.

| Usage | Text Detector | Text Recognizer | Layout Parser | Table Structure Recognizer |
| :--- | :--- | :--- | :--- | :--- |
| CLI (`yomitoku`) | `--td_cfg` | `--tr_cfg` | `--lp_cfg` | `--tsr_cfg` |
| `DocumentAnalyzer(configs=...)` | `["ocr"]["text_detector"]` | `["ocr"]["text_recognizer"]` | `["layout_analyzer"]["layout_parser"]` | `["layout_analyzer"]["table_structure_recognizer"]` |
| `OCR(configs=...)` | `["text_detector"]` | `["text_recognizer"]` | — | — |
| `LayoutAnalyzer(configs=...)` | — | — | `["layout_parser"]` | `["table_structure_recognizer"]` |
| `TableSemanticParser(configs=...)` | `["text_detector"]` | `["text_recognizer"]` | — | — |
| A module on its own | `TextDetector(...)` | `TextRecognizer(...)` | `LayoutParser(...)` | `TableStructureRecognizer(...)` |

### Example: Changing Text Recognizer Settings

Write thresholds and other items in `tr.yaml`.

```yaml
# tr.yaml
rep_min_run_p1: 24     # allow a single character to repeat up to 24 times
data:
  batch_size: 32       # other items under data keep their defaults
```

In the CLI, pass the YAML with `--tr_cfg` and switch features on or off with options.

```bash
yomitoku ${path_data} --tr_cfg tr.yaml --disable-repetition-stop
```

In the Python API, write `path_cfg` and the arguments in the same dictionary.

```python
from yomitoku import OCR, DocumentAnalyzer

# DocumentAnalyzer: write under "ocr"
analyzer = DocumentAnalyzer(
    configs={
        "ocr": {
            "text_recognizer": {
                "path_cfg": "tr.yaml",
                "repetition_stop": False,
                "rec_orientation_fallback": True,
            },
        },
    },
)

# OCR: write directly under configs
ocr = OCR(
    configs={
        "text_recognizer": {
            "path_cfg": "tr.yaml",
            "repetition_stop": False,
        },
    },
)
```

### Notes on Writing YAML

- Writing an item that does not exist causes an error when loading. Check the error message to catch typos in item names.

    ```
    omegaconf.errors.ConfigKeyError: Key 'repetition_stopp' not in 'TextRecognizer...Config'
    ```

- YAML values are applied on top of the defaults of the model selected with `model_name`, so the same YAML can be reused when you switch models.
- Do not change items related to the model's network architecture (such as `encoder` / `decoder`), because they would no longer match the pretrained weights.

### Checking the Applied Settings

Use `log_config()` of each module to log the final settings with arguments and YAML applied. You can also save them to a YAML file with `save_config(path)`.

```python
analyzer.text_recognizer.log_config()   # DocumentAnalyzer
ocr.recognizer.log_config()             # OCR
```

The saved file also contains environment-specific values (such as the absolute path to the charset file). Use it to check the settings; in the YAML you pass to `path_cfg`, write only the items you want to change.

## Text Detector

### input data

```yaml
data:
  # If the number of pixels on the shorter side of the image falls below the specified value, the image will be enlarged to ensure that it meets or exceeds the pixel count set here.
  shortest_size: int 

  # If the number of pixels on the longer side of the image exceeds the specified value, the image will be resized to ensure that it is equal to or less than the pixel count set here.
  limit_size: int 
```

### post process

```yaml
post_process:
  #If the size of the larger side of the detected area falls below the specified value, the area will be removed.
  min_size: int 

  # This is the threshold for the model's prediction score. Pixels with prediction scores below the specified threshold will be treated as background regions.
  thresh: float 

  # The threshold for the model's prediction score is used to treat pixels with prediction scores below the specified threshold as background regions.
  box_thresh: float 

  # The maximum number of detectable text regions.
  max_candidates: int 

  # A parameter to set the size of the margin area for text regions. Larger values increase the margin around text regions, allowing for detection with more whitespace, while smaller values result in tighter detection.
  unclip_ratio: int 

### Visualization

```yaml
visualize:
  # The color of the bounding box for the detected regions.
  color: [B, G, R] 

  # Whether to visualize and render the model's prediction heatmap.
  heatmap: boolean 
```

## Text Recognizer

### maximum text length

```yaml
# The maximum string length that can be predicted. 
max_label_length: int 
```

### input data

```yaml
data:
  # The number of images used for batch processing.
  batch_size: int 
```

### Recognition Orientation Fallback

```yaml
# Whether to retry recognition with a 180-degree rotated ROI image when confidence is low.
rec_orientation_fallback: bool

# The confidence threshold for triggering the fallback.
rec_orientation_fallback_thresh: float
```

The defaults are the same for all models: `rec_orientation_fallback: false` and `rec_orientation_fallback_thresh: 0.75`. Values given with `--enable-rec-orientation-fallback` / `--rec-orientation-fallback-thresh` in the CLI, or `TextRecognizer(rec_orientation_fallback=..., rec_orientation_fallback_thresh=...)` in the Python API, take precedence over the model config.

### Repetition Stop

```yaml
# Whether to stop decoding as soon as a loop of repeated characters or strings is detected, collapsing the repetition to a single unit.
repetition_stop: bool

# The maximum length (in tokens) of a unit regarded as repeating.
rep_period_max: int

# The minimum run length required to suppress a single repeated character.
rep_min_run_p1: int

# The minimum number of repeats required to suppress a multi-character unit.
rep_min_repeats: int
```

The defaults are the same for all models: `repetition_stop: true`, `rep_period_max: 8`, `rep_min_run_p1: 16`, and `rep_min_repeats: 3`. It only works with PyTorch inference and is not applied with ONNX inference (`infer_onnx=True`).

You can disable it with `--disable-repetition-stop` in the CLI or `TextRecognizer(repetition_stop=False)` in the Python API; these take precedence over the model config. See [How to Configure](#how-to-configure) for how to set them from `DocumentAnalyzer` / `OCR`.

### visualization

```yaml
visualize:
  # The path to the font used for visualizing the predicted result strings.
  font: str

  # The color of the font used for visualizing the predicted result strings.
  color: [BGR]

  # The font size of the predicted result strings.
  font_size: int
```

## Layout Parser

### threshold of prediction score

```yaml
# Regions with prediction scores below the specified threshold will be excluded based on the threshold for the model's prediction score.
thresh_score: float 
```

## Table Structure Recognizer

### threshold of prediction score

```yaml
# Regions with prediction scores below the specified threshold will be excluded based on the threshold for the model's prediction score.
thresh_score: float
```
