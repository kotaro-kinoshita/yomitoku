from dataclasses import dataclass, field
from typing import List

from ..constants import ROOT_DIR


@dataclass
class Data:
    num_workers: int = 4
    batch_size: int = 128
    img_size: List[int] = field(default_factory=lambda: [16, 800])


@dataclass
class Encoder:
    patch_size: List[int] = field(default_factory=lambda: [16, 8])
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
class TextRecognizerPARSeqH16LiteDynwConfig:
    """Trial config for the 16px-tall "lite" dynamic-width PARSeq checkpoint.

    A smaller, faster recognizer: 16px input height with a [16, 8] patch
    (one patch row x 100 columns over a 16x800 canvas) and a 368-dim
    encoder/decoder. Trained with the dynamic-width-packing recipe so it is
    robust to narrow canvases, which unlocks accurate dynamic-width batching
    at inference time (``dynamic_width=True``). Weights live in a local
    HF-format directory produced by ``scripts/register_hugging_face_hub.py``
    (with the push_to_hub step disabled), so from_pretrained loads them
    without contacting the Hub.
    """

    # Local HF-format directory (save_pretrained output), loaded by
    # from_pretrained without contacting the Hub.
    hf_hub_repo: str = str(ROOT_DIR + "/local_models/parseq-h16-lite-dynw")
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
