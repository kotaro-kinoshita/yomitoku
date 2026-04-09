from dataclasses import dataclass, field
from typing import List

from ..constants import ROOT_DIR


@dataclass
class Data:
    num_workers: int = 4
    batch_size: int = 128
    img_size: List[int] = field(default_factory=lambda: [32, 400])


@dataclass
class Encoder:
    patch_size: List[int] = field(default_factory=lambda: [8, 16])
    num_heads: int = 8
    embed_dim: int = 368
    mlp_ratio: int = 4
    depth: int = 12


@dataclass
class Decoder:
    embed_dim: int = 368
    num_heads: int = 8
    mlp_ratio: int = 4
    depth: int = 1


@dataclass
class Visualize:
    font: str = str(ROOT_DIR + "/resource/ShipporiMinchoB1-Bold.ttf")
    color: List[int] = field(default_factory=lambda: [0, 0, 255])  # RGB
    font_size: int = 18


@dataclass
class TextRecognizerPARSeqTinyConfig:
    hf_hub_repo: str = "KotaroKinoshita/yomitoku-text-recognizer-parseq-tiny"
    charset: str = str(ROOT_DIR + "/resource/charsetv2.txt")
    num_tokens: int = 7121
    max_label_length: int = 50
    decode_ar: int = 1
    refine_iters: int = 1
    rec_orientation_fallback: bool = False
    rec_orientation_fallback_thresh: float = 0.75

    data: Data = field(default_factory=Data)
    encoder: Encoder = field(default_factory=Encoder)
    decoder: Decoder = field(default_factory=Decoder)

    visualize: Visualize = field(default_factory=Visualize)
