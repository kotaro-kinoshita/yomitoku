from dataclasses import dataclass, field
from typing import List


@dataclass
class Data:
    input_size: List[int] = field(
        default_factory=lambda: [712, 488]
    )  # network input (H, W)


@dataclass
class PostProcess:
    smooth_ctrl: List[int] = field(
        default_factory=lambda: [7, 10]
    )  # B-spline control points (u, v); [] = off
    snap: bool = True  # snap the grid outline to the paper mask (SDF head)
    snap_bias: float = -0.06  # px (network input) subtracted from the SDF zero level
    snap_max_dist: float = 10.0  # search range px (network input)
    snap_side_ctrl: int = 12  # control points of the snapped sides
    snap_decay: float = 3.0  # nodes over which the outline correction fades inward
    scale: float = 1.5  # dewarped page size relative to its size in the photo
    focal_diag: float = 0.6  # camera focal / photo diagonal (phone ~26 mm equivalent), for the output aspect
    fill_background: bool = (
        True  # paint what lies outside the paper or the photo with the paper colour
    )
    fill_margin: float = 0.0  # SDF level of the paper edge (SDF clipped to ±1, ~0.25 per network-input px)


@dataclass
class Reject:
    """A dewarp is rejected (the analysis continues on the input image) when any check fails. Thresholds set on
    real photos (99 receipts, 100 handwritten notebooks, 13 documents): every handwritten reject was visibly broken
    (mostly the neighbouring page of a spread folded into the grid); good pages stay below 0.07 flipped cells."""

    min_mask_iou: float = 0.85  # IoU of the grid outline and the paper mask
    max_flipped_cells: float = (
        0.08  # share of grid cells turned over (a neighbouring page folded in)
    )
    max_spacing_ratio: float = 6.0  # longest / median grid edge
    max_outside: float = 0.5  # share of grid nodes far outside the photo (> 5 % of its size); pages cut by the frame dewarp fine


@dataclass
class DewarperUVDocV1Config:
    hf_hub_repo: str = "MLism/yomitoku-dewarper-uvdoc-v1"
    width: int = 32
    kernel: int = 5
    sdf: bool = True
    data: Data = field(default_factory=Data)
    postprocess: PostProcess = field(default_factory=PostProcess)
    reject: Reject = field(default_factory=Reject)
