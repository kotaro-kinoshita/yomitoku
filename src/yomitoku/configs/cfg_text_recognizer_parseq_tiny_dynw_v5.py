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
    dynamic_width: bool = True
    batch_bucketing: bool = True
    resize_policy: str = "fit"
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
class TextRecognizerPARSeqTinyDynwV5Config:
    """Config for the charset-v3 lite dynamic-width PARSeq recognizer
    (parseq-tiny-dynw-v5).

    Same architecture as ``parseq-tiny-dynw-v4`` (32x800 canvas, [4, 8] patch,
    192-dim encoder/decoder, 6 heads), trained on charset v3 (7519 chars,
    2026-07-31 revision) WITHOUT uniform NFKC normalization of the labels. Accordingly, inference
    must not NFKC-normalize predictions either; instead an explicit
    per-character replacement table (character_post_expand_table_v3) expands
    the few composite glyphs (e.g. ℡ -> TEL) in post-processing.

    """

    hf_hub_repo: str = "KotaroKinoshita/yomitoku-text-recognizer-parseq-tiny-dynw-v5"
    charset: str = str(ROOT_DIR + "/resource/charsetv3.txt")
    num_tokens: int = 7522
    max_label_length: int = 100
    decode_ar: int = 1
    refine_iters: int = 1
    # Preserve legitimate receipt masks and leader lines up to a 16-character
    # constant run, while stopping genuine multi-token loops after 3 repeats.
    # PARSeq's repeat detector excludes constant units from period > 1 so the
    # two thresholds can be tuned independently.
    repetition_stop: bool = True
    rep_period_max: int = 8
    rep_min_run_p1: int = 16
    rep_min_repeats: int = 3
    rec_orientation_fallback: bool = False
    rec_orientation_fallback_thresh: float = 0.75

    # charset v3 keeps compatibility characters verbatim, so the uniform
    # NFKC pass must be skipped; the table below replaces it.
    nfkc_normalize: bool = False
    char_replace_table: Optional[str] = str(
        ROOT_DIR + "/resource/character_post_expand_table_v3.csv"
    )

    data: Data = field(default_factory=Data)
    encoder: Encoder = field(default_factory=Encoder)
    decoder: Decoder = field(default_factory=Decoder)

    visualize: Visualize = field(default_factory=Visualize)
