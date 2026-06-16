from dataclasses import dataclass, field
from typing import List

from ..constants import ROOT_DIR


@dataclass
class Data:
    num_workers: int = 4
    batch_size: int = 128
    img_size: List[int] = field(default_factory=lambda: [32, 800])


@dataclass
class Encoder:
    patch_size: List[int] = field(default_factory=lambda: [8, 8])
    num_heads: int = 8
    embed_dim: int = 512
    mlp_ratio: int = 4
    depth: int = 12


@dataclass
class Decoder:
    embed_dim: int = 512
    num_heads: int = 8
    mlp_ratio: int = 4
    depth: int = 1


@dataclass
class Visualize:
    font: str = str(ROOT_DIR + "/resource/ShipporiMinchoB1-Bold.ttf")
    color: List[int] = field(default_factory=lambda: [0, 0, 255])  # RGB
    font_size: int = 18


@dataclass
class TextRecognizerPARSeqMiddleV4DynwConfig:
    """Trial config for the d512 ("middle-v4") dynamic-width PARSeq checkpoint.

    Same overall PARSeq architecture as parseq-large-v4_1, but with a 512-dim
    encoder/decoder (the "middle" size) instead of 768. Weights live in a
    local HF-format directory produced by
    ``scripts/register_hugging_face_hub.py`` (with the push_to_hub step
    disabled) from the dynw-middle checkpoint, so from_pretrained loads them
    without any download. The checkpoint was fine-tuned with the dynw recipe
    (width bucketing + random trailing margin) so it is robust to narrow
    canvases, which unlocks accurate dynamic-width batching at inference time.
    """

    # Local HF-format directory (save_pretrained output), loaded by
    # from_pretrained without contacting the Hub.
    hf_hub_repo: str = str(ROOT_DIR + "/local_models/parseq-middle-v4-dynw")
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
