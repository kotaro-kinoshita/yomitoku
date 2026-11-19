from dataclasses import dataclass, field

from .cfg_dewarper_pagenet_v1 import Data, Detect, Filter, PostProcess


@dataclass
class Snap:
    """Each page's outline is moved to the paper SDF's zero level (same post-process as the single-page dewarper,
    `utils.dewarp.snap_grid`). The SDF is + on any page, so only the outline against the background moves; a spread's
    binding and edges lying on another page stay as predicted. A snap that folds a cell or shrinks the smallest cell
    by more than 20 % is dropped."""

    enabled: bool = True
    bias: float = -0.06  # px (network input) subtracted from the SDF zero level
    max_dist: float = 10.0  # search range px (network input)
    side_ctrl: int = 12  # control points of the snapped sides
    decay: float = 3.0  # nodes over which the outline correction fades inward


@dataclass
class Flat:
    """Flat pages (PDF renders, flatbed scans, a sheet shot from straight above): a page the 3D head finds planar
    (out-of-plane / in-plane spread below `max_flatness`) and facing the camera (plane normal within `max_tilt`
    degrees of the camera axis) is not unwarped through its grid; the photo is only rotated (one affine warp, no
    shear or per-axis scale) so that the page is level, and cropped to the page. The grid's slight keystone and uneven
    spacing on such pages would otherwise bend straight lines. Thresholds set on 7.8k synthetic pages: 94 % of the
    flat / scanned pages pass, 3 % of the others (which are then within 0.5 px of their grid unwarp); every PDF and
    flatbed scan of the real sets passes, 0-10 % of photos of curved paper.

    The angle is the rectangle fitted to the grid, refined (`deskew`) from the page's text lines and rules or, when
    they give no confident angle (nearly blank pages), from the paper edges against the background."""

    enabled: bool = True
    max_flatness: float = 0.01
    max_tilt: float = 2.0  # degrees
    deskew: bool = True
    deskew_max_deg: float = 1.5  # search range of the content-based refinement
    edge_max_deg: float = 2.0  # search range of the paper-edge refinement
    scale: float = 1.0  # output size relative to the page in the photo (one resampling only: keep the resolution)


@dataclass
class DewarperPageNetV2Config:
    """pagenet_v1 retrained on more data for its weak spots (pages with little text, white paper on a white desk,
    receipts) with the paper SDF head: the page outlines are snapped to the paper edge and flat pages are only
    rotated."""

    hf_hub_repo: str = "KotaroKinoshita/yomitoku-dewarper-pagenet-v2"
    width: int = 32
    kernel: int = 5
    roi_channels: int = 128
    sdf: bool = True
    data: Data = field(default_factory=Data)
    detect: Detect = field(default_factory=Detect)
    filter: Filter = field(default_factory=Filter)
    snap: Snap = field(default_factory=Snap)
    flat: Flat = field(default_factory=Flat)
    postprocess: PostProcess = field(default_factory=PostProcess)
