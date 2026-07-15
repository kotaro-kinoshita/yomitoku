from dataclasses import dataclass, field
from typing import List, Optional

from ..constants import ROOT_DIR


@dataclass
class Data:
    num_workers: int = 4
    batch_size: int = 10
    img_size: List[int] = field(default_factory=lambda: [32, 800])
    # Dynamic batch size for the dynamic-width path: batches are filled until
    # (len(batch) * batch_max_width) exceeds width_budget, so narrow crops form
    # large batches and wide crops small ones (constant padded pixels/step).
    # Default 8000 = batch_size(10) * img_width(800), i.e. full-width parity
    # with the fixed-batch worst case. Requires dynamic_width=True (and pairs
    # with batch_bucketing so batches stay width-homogeneous).
    width_budget: int = 8000
    # Optional hard cap on batch size to bound decode-time memory on many
    # very narrow crops. None = no cap.
    max_batch_size: Optional[int] = 64


@dataclass
class Encoder:
    patch_size: List[int] = field(default_factory=lambda: [4, 8])
    num_heads: int = 6
    embed_dim: int = 192
    mlp_ratio: int = 4
    depth: int = 12


@dataclass
class Decoder:
    embed_dim: int = 192
    num_heads: int = 6
    mlp_ratio: int = 4
    depth: int = 1


@dataclass
class Visualize:
    font: str = str(ROOT_DIR + "/resource/ShipporiMinchoB1-Bold.ttf")
    color: List[int] = field(default_factory=lambda: [0, 0, 255])  # RGB
    font_size: int = 18


@dataclass
class TextRecognizerPARSeqTinyDynwV4Config:
    """Config for the lite dynamic-width PARSeq recognizer (parseq-tiny-dynw-v4).

    A compact recognizer: 32px input height with a [4, 8] patch (8 patch rows
    x 100 columns over a 32x800 canvas), a 192-dim encoder/decoder, and 6
    attention heads. Trained with the dynamic-width recipe (width bucketing +
    trailing margin) so it stays accurate on narrow canvases, which unlocks
    dynamic-width batching at inference time (``dynamic_width=True``, paired
    with ``batch_bucketing=True``). This is the recognizer selected by the CLI
    ``--lite`` option for fast CPU inference.
    """

    hf_hub_repo: str = "KotaroKinoshita/yomitoku-text-recognizer-parseq-tiny-dynw-v4"
    charset: str = str(ROOT_DIR + "/resource/charsetv2.txt")
    num_tokens: int = 7121
    max_label_length: int = 100
    decode_ar: int = 1
    refine_iters: int = 1
    rec_orientation_fallback: bool = False
    rec_orientation_fallback_thresh: float = 0.75

    data: Data = field(default_factory=Data)
    encoder: Encoder = field(default_factory=Encoder)
    decoder: Decoder = field(default_factory=Decoder)

    visualize: Visualize = field(default_factory=Visualize)
