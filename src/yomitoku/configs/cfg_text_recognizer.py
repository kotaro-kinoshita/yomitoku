from dataclasses import dataclass, field
from typing import List


@dataclass
class Data:
    num_workers: int = 4
    batch_size: int = 128
    img_size: List[int] = field(default_factory=lambda: [32, 800])


@dataclass
class Parseq:
    max_label_length: int = 100
    patch_size: List[int] = field(default_factory=lambda: [8, 8])
    embed_dim: int = 512
    enc_num_heads: int = 8
    enc_mlp_ratio: int = 4
    enc_depth: int = 12
    dec_num_heads: int = 8
    dec_mlp_ratio: int = 4
    dec_depth: int = 1
    decode_ar: bool = True
    refine_iters: int = 1
    dropout: float = 0.1


@dataclass
class Visualize:
    font: str = "resource/MPLUS1p-Medium.ttf"
    color: List[int] = field(default_factory=lambda: [0, 0, 255])  # RGB
    font_size: int = 18


@dataclass
class TextRecognizerConfig:
    hf_hub_repo: str = (
        "KotaroKinoshita/yomitoku-text-recognizer-parseq-open-beta"
    )
    charset: str = "resource/charset.txt"
    data: Data = Data()
    parseq: Parseq = Parseq()
    visualize: Visualize = Visualize()
