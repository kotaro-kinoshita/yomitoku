from dataclasses import dataclass, field
from typing import List


@dataclass
class Data:
    input_size: List[int] = field(
        default_factory=lambda: [712, 488]
    )  # network input (H, W)


@dataclass
class Detect:
    thresh: float = 0.3  # page-center heatmap score
    max_pages: int = 16
    refine: int = 1  # extra grid predictions on the corners of the previous grid
    dedup: float = (
        0.06  # pages whose corners lie closer (mean, normalized coords) are one page
    )


@dataclass
class Filter:
    """Detected pages dropped before dewarping, in this order. Thresholds set on real photos (509 documents, 100
    handwritten notebooks, 20 business cards, 17 documents, 354 receipts): the best page of every photo has
    cell_min >= 0.37, inside >= 0.44 and seen >= 0.46; cut-off neighbouring pages of a spread lie at inside 0.04-0.3."""

    min_cell: float = 0.25  # collapsed grid: 2nd-percentile / median cell area below this (or any folded cell)
    nest: float = 0.8  # a page lying this much inside a larger page is dropped, whatever the scores
    min_inside: float = 0.5  # cut off by the frame: share of grid nodes inside the photo (best page always kept)
    min_seen: float = (
        0.3  # cut off: share of grid nodes predicted seen (best page always kept)
    )


@dataclass
class PostProcess:
    cut_seen: bool = (
        True  # dewarp only the seen part of each page (not the parts outside the photo)
    )
    scale: float = 1.5  # dewarped page size relative to its size in the photo
    focal_diag: float = 0.6  # camera focal / photo diagonal (phone ~26 mm equivalent), for the output aspect
    fill_background: bool = (
        True  # paint what lies outside the photo with the paper colour
    )


@dataclass
class DewarperPageNetV1Config:
    hf_hub_repo: str = "MLism/yomitoku-dewarper-pagenet-v1"
    width: int = 32
    kernel: int = 5
    roi_channels: int = 128
    data: Data = field(default_factory=Data)
    detect: Detect = field(default_factory=Detect)
    filter: Filter = field(default_factory=Filter)
    postprocess: PostProcess = field(default_factory=PostProcess)
