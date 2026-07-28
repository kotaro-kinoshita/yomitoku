from dataclasses import dataclass, field
from typing import List


@dataclass
class Data:
    # 960 検証用: 入力画像サイズを 960 に拡大
    img_size: List[int] = field(default_factory=lambda: [960, 960])


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
    # 6 classes: table, cell, header, empty, kv_item, grid
    num_classes: int = 6
    feat_channels: List[int] = field(default_factory=lambda: [256, 256, 256])
    feat_strides: List[int] = field(default_factory=lambda: [8, 16, 32])
    hidden_dim: int = 256
    num_levels: int = 3

    num_layers: int = 6
    # 学習時 (rtdetrv2_pytorch/configs/table_semantic_parser) は 900。
    # learn_query_content=False のため、推論時に増やしても重みの shape は不変
    # (encoder proposal の top-k 選択数が増えるだけ)。密なテーブル (最大 ~1000
    # セル) で検出上限に達しないよう余裕を持たせる。
    num_queries: int = 1500

    num_denoising: int = 100
    label_noise_ratio: float = 0.5
    box_noise_scale: float = 1.0
    # 960 検証用: 入力画像サイズに合わせて拡大
    eval_spatial_size: List[int] = field(default_factory=lambda: [960, 960])

    eval_idx: int = -1

    num_points: List[int] = field(default_factory=lambda: [4, 4, 4])
    cross_attn_method: str = "default"
    query_select_method: str = "default"


@dataclass
class TableCellParserRTDETRv2Beta960Config:
    hf_hub_repo: str = "KotaroKinoshita/yomitoku-cell-detector-rtdtrv2-beta"
    # ローカルの学習済みチェックポイントを直接読み込む場合のパス。
    # 設定されている場合は hf_hub_repo より優先される（960 入力サイズの beta 検証用）。
    # TODO: 検証後、重みを HF Hub に上げて hf_hub_repo に切り替える。
    weights_path: str = (
        "/home/user/Projects/rtdetrv2_pytorch/output/table_semantic_parser_960_aug/best.pth"
    )
    # チェックポイント内で使用する重みキー ("model" or "ema")
    weights_key: str = "ema"
    thresh_score: float = 0.5
    data: Data = field(default_factory=Data)
    PResNet: BackBone = field(default_factory=BackBone)
    HybridEncoder: Encoder = field(default_factory=Encoder)
    RTDETRTransformerv2: Decoder = field(default_factory=Decoder)

    # 0-indexed 6クラス (rtdetrv2_pytorch/dataset/table_crops と一致)
    category: List[str] = field(
        default_factory=lambda: [
            "table",
            "cell",
            "header",
            "empty",
            "kv_item",
            "grid",
        ]
    )
