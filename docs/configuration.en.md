# Model Config

The configurable parameters for each Model are explained.

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
