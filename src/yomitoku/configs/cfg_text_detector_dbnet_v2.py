from dataclasses import dataclass, field
from typing import List


@dataclass
class BackBone:
    name: str = "resnet50"
    dilation: bool = True


@dataclass
class Decoder:
    in_channels: list[int] = field(default_factory=lambda: [256, 512, 1024, 2048])
    hidden_dim: int = 256
    adaptive: bool = True
    serial: bool = True
    smooth: bool = False
    k: int = 50


@dataclass
class Data:
    shortest_size: int = 1280
    limit_size: int = 1600


@dataclass
class PostProcess:
    min_size: int = 2
    thresh: float = 0.2
    box_thresh: float = 0.5
    max_candidates: int = 1500
    unclip_ratio: float = 3.0


@dataclass
class Visualize:
    color: List[int] = field(default_factory=lambda: [0, 255, 0])
    heatmap: bool = False


@dataclass
class TextDetectorDBNetV2Config:
    hf_hub_repo: str = "KotaroKinoshita/yomitoku-text-detector-dbnet-v2"
    backbone: BackBone = field(default_factory=BackBone)
    decoder: Decoder = field(default_factory=Decoder)
    data: Data = field(default_factory=Data)
    post_process: PostProcess = field(default_factory=PostProcess)
    visualize: Visualize = field(default_factory=Visualize)
