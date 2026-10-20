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
| `TableSemanticParser(configs=...)` | `["text_detector"]` | `["text_recognizer"]` | `["table_detector"]` | — |
| A module on its own | `TextDetector(...)` | `TextRecognizer(...)` | `LayoutParser(...)` | `TableStructureRecognizer(...)` |

Settings for the cell detection model of `TableSemanticParser` (Table Cell Parser) go under `configs["table_cell_parser"]`.

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

The model that detects text regions. You can tune the input image size and the post-processing of detections.

| Item | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data.shortest_size` | `int` | `1280` | If the shorter side of the image is smaller than this value, the image is enlarged so that the shorter side matches it. Increase it when small text is missed. |
| `data.limit_size` | `int` | `1600` | If the longer side of the image exceeds this value, the image is shrunk so that the longer side fits within it. Larger values help detect fine text but increase processing time and memory usage. |
| `post_process.thresh` | `float` | Varies by model (see below) | The threshold for binarizing the predicted heatmap. Pixels above this value are treated as text. Lowering it picks up faint text; raising it reduces noise. |
| `post_process.box_thresh` | `float` | Varies by model (see below) | Regions whose average score is below this value are discarded. |
| `post_process.min_size` | `int` | `2` | Regions whose shorter side is below this value are discarded. |
| `post_process.max_candidates` | `int` | `1500` | The maximum number of regions processed per page. Increase it when text is missed on pages with a very large amount of text. |
| `post_process.unclip_ratio` | `float` | Varies by model (see below) | How much the detected region is expanded outward. Larger values give looser boxes with more margin; smaller values give tighter boxes. |
| `visualize.heatmap` | `bool` | `false` | Whether to overlay the predicted heatmap on the visualization. |
| `visualize.color` | `[B, G, R]` | `[0, 255, 0]` | The color of the detection boxes in the visualization. |

Defaults by model:

| Model | `post_process.thresh` | `post_process.box_thresh` | `post_process.unclip_ratio` |
| :--- | :--- | :--- | :--- |
| `dbnet` | `0.15` | `0.5` | `7.0` |
| `dbnetv2` | `0.2` | `0.5` | `5.0` |
| `dbnetv2_1` | `0.3` | `0.4` | `3.5` |

`backbone.*` / `decoder.*` describe the network architecture and must match the pretrained weights, so they cannot be changed.

## Text Recognizer

The model that recognizes the text in each detected region. Items marked ✔ in the "Argument" column can also be passed as constructor arguments or in the `configs` dictionary, and those values take precedence over YAML.

### Inference and Batching

| Item | Type | Default | Argument | Description |
| :--- | :--- | :--- | :---: | :--- |
| `data.batch_size` | `int` | Varies by model (see below) |  | The number of text regions processed in one inference (unless `width_budget` is used). Larger values are faster on GPU but use more memory. |
| `data.dynamic_width` | `bool` | Varies by model (see below) | ✔ | Infers each crop at its content width instead of padding to a fixed width. Enable only for models that support it. Always disabled with ONNX inference. |
| `data.batch_bucketing` | `bool` | Varies by model (see below) | ✔ | Groups crops of similar width into the same batch. Effective together with `dynamic_width`. |
| `data.width_budget` | `int` | Varies by model (see below) |  | With `dynamic_width`, batches are split so that "number of crops × widest crop" stays within this value. Lower it if GPU memory runs out. |
| `data.max_batch_size` | `int` | Varies by model (see below) |  | The maximum number of crops per batch when `width_budget` is used. |
| `data.num_workers` | `int` | `8` |  | The number of threads used to preprocess the crops. |
| `decode_ar` | `int` | `1` |  | `1` uses autoregressive decoding and `0` non-autoregressive decoding. `0` is faster but may change results and lower accuracy. |
| `refine_iters` | `int` | `1` |  | The number of refinement iterations after decoding. `0` makes processing lighter. |

Defaults by model (`—` means the item does not exist in that model's config and cannot be set):

| Model | `data.batch_size` | `data.dynamic_width` | `data.batch_bucketing` | `data.width_budget` | `data.max_batch_size` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `parseq` | `128` | `false` | `false` | — | — |
| `parseqv2` | `128` | `false` | `false` | — | — |
| `parseq-small` | `128` | `false` | `false` | — | — |
| `parseq-tiny` | `128` | `false` | `false` | — | — |
| `parseq-large-v4_1` | `128` | `false` | `false` | — | — |
| `parseq-tiny-dynw-v4` | `10` | `true` | `true` | `8000` | `64` |
| `parseq-tiny-dynw-v5` | `10` | `true` | `true` | `8000` | `64` |
| `parseq-middle-dynw-v5` | `10` | `true` | `true` | `8000` | `64` |

### Repetition Stop

| Item | Type | Default | Argument | Description |
| :--- | :--- | :--- | :---: | :--- |
| `repetition_stop` | `bool` | `true` | ✔ | Stops decoding when a loop of repeated characters or strings is detected and collapses the repetition to a single unit. Works only with PyTorch inference, not ONNX. Disable it with `--disable-repetition-stop` in the CLI. |
| `rep_period_max` | `int` | `8` |  | The maximum length (in tokens) of a unit regarded as repeating. |
| `rep_min_run_p1` | `int` | `16` |  | The minimum run length to suppress a single repeated character. Increase it to keep long runs such as leader dots (`……`). |
| `rep_min_repeats` | `int` | `3` |  | The minimum number of repeats to suppress a multi-character unit. |

### Recognition Orientation Fallback

| Item | Type | Default | Argument | Description |
| :--- | :--- | :--- | :---: | :--- |
| `rec_orientation_fallback` | `bool` | `false` | ✔ | Re-recognizes low-confidence lines rotated by 180 degrees and keeps the result with higher confidence. Enable it with `--enable-rec-orientation-fallback` in the CLI. |
| `rec_orientation_fallback_thresh` | `float` | `0.75` | ✔ | The confidence threshold for triggering the fallback. Set it with `--rec-orientation-fallback-thresh` in the CLI. |

### Visualization

| Item | Type | Default | Argument | Description |
| :--- | :--- | :--- | :---: | :--- |
| `visualize.font` | `str` | Varies by model |  | Path to the font file (.ttf) used to draw recognized text in the visualization. |
| `visualize.color` | `[B, G, R]` | `[0, 0, 255]` |  | The color of the recognized text (used in the Text Recognizer's own visualization and the Table Semantic Parser visualization). |
| `visualize.font_size` | `int` | `18` |  | The font size of the recognized text (used in the Text Recognizer's own visualization and the Table Semantic Parser visualization). |

### Items Matched to the Model

The following items can be changed, but they are matched to how the model was trained. Changing them lowers accuracy, so normally keep the model's values.

- `data.resize_policy`: how crops are resized (`fit` also enlarges small crops, `downscale` only shrinks)
- `data.trailing_margin`: the margin added at the end of a line in dynamic-width inference
- `nfkc_normalize`: whether NFKC normalization is applied to recognized text
- `char_replace_table`: path to the table that replaces characters in recognized text

`data.trailing_margin` / `nfkc_normalize` / `char_replace_table` can be set only for models whose config has these items.

### Items That Cannot Be Changed

The following items must match the pretrained weights and cannot be changed. Changing them causes an error when loading the model or at inference.

- `max_label_length` (maximum number of characters)
- `data.img_size` (input image size)
- `charset` / `num_tokens` (character set)
- `encoder.*` / `decoder.*` (network architecture)
- `hf_hub_repo` (where the weights are downloaded from)

In addition, `num_parallel_batches` (the number of batches inferred in parallel) and `source_downscale` (whether to shrink high-resolution images before cropping) can be set only as `TextRecognizer` arguments.

## Layout Parser / Table Structure Recognizer / Table Cell Parser

For the layout and table models, the tunable item is the prediction score threshold `thresh_score`. Raising it keeps only confident regions; lowering it reduces misses but increases false detections.

| Model | Used for | Default `thresh_score` |
| :--- | :--- | :--- |
| Layout Parser | Layout analysis in `DocumentAnalyzer` / `LayoutAnalyzer` | `0.5` |
| Table Structure Recognizer | Table structure recognition in `DocumentAnalyzer` / `LayoutAnalyzer` | `0.4` |
| Table Cell Parser | Cell detection in `TableSemanticParser` (`configs["table_cell_parser"]`) | `0.5` |

Table detection in `TableSemanticParser` (`configs["table_detector"]`) uses the Layout Parser, so it is tuned with the same `thresh_score`.

`data.img_size`, `category` and the network architecture (`PResNet.*` / `HybridEncoder.*` / `RTDETRTransformerv2.*`) are model-specific and cannot be changed. Changing `data.img_size` causes an error at inference.
