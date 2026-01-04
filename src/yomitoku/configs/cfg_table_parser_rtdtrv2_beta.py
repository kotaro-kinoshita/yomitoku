from dataclasses import dataclass, field
from typing import List


@dataclass
class Data:
    img_size: List[int] = field(default_factory=lambda: [640, 640])


@dataclass
class BackBone:
    depth: int = 50
    variant: str = "d"
    freeze_at: int = 0
    return_idx: List[int] = field(default_factory=lambda: [1, 2, 3])
    num_stages: int = 4
    freeze_norm: bool = True


@dataclass
class Encoder:
    in_channels: List[int] = field(default_factory=lambda: [512, 1024, 2048])
    feat_strides: List[int] = field(default_factory=lambda: [8, 16, 32])

    # intra
    hidden_dim: int = 256
    use_encoder_idx: List[int] = field(default_factory=lambda: [2])
    num_encoder_layers: int = 1
    nhead: int = 8
    dim_feedforward: int = 1024
    dropout: float = 0.0
    enc_act: str = "gelu"

    # cross
    expansion: float = 1.0
    depth_mult: int = 1
    act: str = "silu"


@dataclass
class Decoder:
    num_classes: int = 8
    feat_channels: List[int] = field(default_factory=lambda: [256, 256, 256])
    feat_strides: List[int] = field(default_factory=lambda: [8, 16, 32])
    hidden_dim: int = 256
    num_levels: int = 3

    num_layers: int = 6
    num_queries: int = 1000

    num_denoising: int = 100
    label_noise_ratio: float = 0.5
    box_noise_scale: float = 1.0
    eval_spatial_size: List[int] = field(default_factory=lambda: [640, 640])

    eval_idx: int = -1

    num_points: List[int] = field(default_factory=lambda: [6, 6, 6])
    cross_attn_method: str = "default"
    query_select_method: str = "default"


@dataclass
class TableParserRTDETRv2BetaConfig:
    # hf_hub_repo: str = "weights/table_parser_phase2_20251228"
    hf_hub_repo: str = "weights/table_parser_phase2_20260104"
    thresh_score: float = 0.5
    data: Data = field(default_factory=Data)
    PResNet: BackBone = field(default_factory=BackBone)
    HybridEncoder: Encoder = field(default_factory=Encoder)
    RTDETRTransformerv2: Decoder = field(default_factory=Decoder)

    category: List[str] = field(
        default_factory=lambda: [
            "table",
            "cell",
            "form",
            "header",
            "check",
            "select",
            "group",
            "empty",
        ]
    )
