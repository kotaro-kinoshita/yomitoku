from dataclasses import dataclass, field
from typing import List, Optional

from ..constants import ROOT_DIR


@dataclass
class Data:
    num_workers: int = 4
    batch_size: int = 10
    img_size: List[int] = field(default_factory=lambda: [32, 800])
    dynamic_width: bool = True
    batch_bucketing: bool = True
    resize_policy: str = "fit"
    width_budget: int = 8000
    max_batch_size: Optional[int] = 64


@dataclass
class Encoder:
    patch_size: List[int] = field(default_factory=lambda: [4, 8])
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
    color: List[int] = field(default_factory=lambda: [0, 0, 255])
    font_size: int = 18


@dataclass
class TextRecognizerPARSeqMiddleDynwV5Config:
    """Charset-v3, non-NFKC, dynamic-width PARSeq middle recognizer."""

    hf_hub_repo: str = "KotaroKinoshita/yomitoku-text-recognizer-parseq-middle-dynw-v5"
    charset: str = str(ROOT_DIR + "/resource/charsetv3.txt")
    num_tokens: int = 7522
    max_label_length: int = 100
    decode_ar: int = 1
    refine_iters: int = 1
    repetition_stop: bool = True
    rep_period_max: int = 8
    rep_min_run_p1: int = 16
    rep_min_repeats: int = 3
    rec_orientation_fallback: bool = False
    rec_orientation_fallback_thresh: float = 0.75
    nfkc_normalize: bool = False
    char_replace_table: Optional[str] = str(
        ROOT_DIR + "/resource/character_post_expand_table_v3.csv"
    )

    data: Data = field(default_factory=Data)
    encoder: Encoder = field(default_factory=Encoder)
    decoder: Decoder = field(default_factory=Decoder)
    visualize: Visualize = field(default_factory=Visualize)
