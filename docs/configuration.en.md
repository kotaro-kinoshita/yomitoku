# Model Config

This page explains the configurable parameters for each model.

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

You can disable it with `--disable-repetition-stop` in the CLI or `TextRecognizer(repetition_stop=False)` in the Python API; these take precedence over the model config. For `DocumentAnalyzer` / `OCR`, pass `configs={"ocr": {"text_recognizer": {"repetition_stop": False}}}`.

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
