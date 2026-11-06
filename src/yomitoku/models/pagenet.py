"""Multi-page geometric correction network, ported from docwarp-synth (`docwarp_synth/dewarp/pages.py`): finds every
page in a photo (the two pages of a spread, receipts side by side, ...) and predicts each page's grid.

One backbone pass over the photo (DewarpNet's encoder and bridges, stride 4 / 8 / 16 features), then
  - page detection (CenterNet-style, stride 8): a heatmap of page centers and, at each center, the offsets to the
    page's four grid corners (u, v) = (0, 0), (1, 0), (1, 1), (0, 1); the corners may lie outside the photo;
  - per page (ROI): features of the three levels and the photo are sampled on the quad of the four corners (expanded
    by ROI_MARGIN on each side) into a fixed ROI map; a small dilated CNN predicts, at each of the 45 x 31 grid
    nodes, the offset from the quad's bilinear position (in units of the quad size), the 3D position and whether the
    node is seen (in the frame and not hidden by another page).
"""

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

from .dewarpnet import DewarpNet, _conv_bn_act

GRID_HW = (45, 31)  # grid nodes (v, u) of a page
DET_STRIDE = 8
ROI_HW = (96, 66)  # ROI feature map (v, u)
ROI_MARGIN = 0.12  # ROI covers u, v in [-m, 1 + m] of the quad
CORNER_UNIT = 64.0  # corner offsets are regressed in units of 64 input px


def quad_points(
    corners: torch.Tensor, u: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Bilinear map of the quad (N, 4, 2) [c00, c10, c11, c01] at (u, v) (broadcast to (..., H, W)) -> (N, H, W, 2)."""
    c00, c10, c11, c01 = [corners[:, i, None, None] for i in range(4)]
    u, v = u[None, ..., None], v[None, ..., None]
    return (1 - u) * (1 - v) * c00 + u * (1 - v) * c10 + u * v * c11 + (1 - u) * v * c01


def _uv(h: int, w: int, lo: float, hi: float, dev):
    v, u = torch.meshgrid(
        torch.linspace(lo, hi, h, device=dev),
        torch.linspace(lo, hi, w, device=dev),
        indexing="ij",
    )
    return u, v


class ResBlock(nn.Module):
    def __init__(self, c: int, d: int):
        super().__init__()
        self.a = _conv_bn_act(c, c, d)
        self.b = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, d, d, bias=False), nn.BatchNorm2d(c)
        )

    def forward(self, x):
        return F.relu(x + self.b(self.a(x)))


class PageNet(nn.Module, PyTorchModelHubMixin):
    """Page detection + per-page grid regression. Input (B, 3, H, W) RGB in [0, 1]; `forward` returns the heatmap
    logits (B, 1, H/8, W/8), the corner offsets (B, 8, H/8, W/8) and the backbone features; `roi` predicts the
    grids of pages given by their corner quads."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        width = cfg.width
        base = DewarpNet(SimpleNamespace(width=width, kernel=cfg.kernel, sdf=False))
        self.resnet_head, self.resnet_down = base.resnet_head, base.resnet_down
        for i in range(1, 7):
            setattr(self, f"bridge_{i}", getattr(base, f"bridge_{i}"))
        self.bridge_concat = base.bridge_concat
        c4, c8, c16 = width, width * 2, width * 4
        self.det_neck = nn.Sequential(
            _conv_bn_act(c16 + c8, 128, 1),
            _conv_bn_act(128, 128, 2),
            _conv_bn_act(128, 128, 4),
        )
        self.det_heat = nn.Sequential(_conv_bn_act(128, 64, 1), nn.Conv2d(64, 1, 1))
        self.det_corner = nn.Sequential(_conv_bn_act(128, 64, 1), nn.Conv2d(64, 8, 1))
        roi_c = cfg.roi_channels
        self.roi_in = _conv_bn_act(c4 + c8 + c16 + 3 + 5, roi_c, 1)
        self.roi_body = nn.Sequential(
            ResBlock(roi_c, 1),
            ResBlock(roi_c, 2),
            ResBlock(roi_c, 4),
            ResBlock(roi_c, 8),
            nn.Conv2d(roi_c, roi_c, 3, 2, 1, bias=False),
            nn.BatchNorm2d(roi_c),
            nn.ReLU(inplace=True),
            ResBlock(roi_c, 1),
            ResBlock(roi_c, 2),
            ResBlock(roi_c, 4),
        )
        self.roi_out = nn.Conv2d(roi_c, 2 + 3 + 1, 3, 1, 1)

    def features(self, x):
        e = self.resnet_down
        f4 = e.layer1(self.resnet_head(x))
        f8 = e.layer2(f4)
        f = e.layer3(f8)
        f16 = self.bridge_concat(
            torch.cat([getattr(self, f"bridge_{i}")(f) for i in range(1, 7)], 1)
        )
        return f4, f8, f16

    def detect(self, f8, f16):
        x = self.det_neck(
            torch.cat(
                [
                    F.interpolate(
                        f16, size=f8.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    f8,
                ],
                1,
            )
        )
        return self.det_heat(x), self.det_corner(x)

    def roi(self, feats, image, img_idx, corners):
        """Pages given by quads `corners` (N, 4, 2) in normalized image coords [-1, 1] of photo `img_idx` (N,) ->
        grid2d (N, 2, 45, 31) [-1, 1], grid3d (N, 3, 45, 31), seen logits (N, 1, 45, 31)."""
        N = len(corners)
        hr, wr = ROI_HW
        m = ROI_MARGIN
        u, v = _uv(hr, wr, -m, 1 + m, corners.device)
        pts = quad_points(corners, u, v)
        cols = [
            F.grid_sample(
                f[img_idx].float(),
                pts,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            for f in feats + (image,)
        ]
        inside = (pts.abs() <= 1).all(-1, keepdim=True).float()
        coord = torch.cat(
            [
                u.expand(N, -1, -1)[..., None],
                v.expand(N, -1, -1)[..., None],
                pts,
                inside,
            ],
            -1,
        )
        x = torch.cat(cols + [coord.permute(0, 3, 1, 2)], 1)
        y = self.roi_out(self.roi_body(self.roi_in(x)))
        gh, gw = GRID_HW
        gu, gv = _uv(gh, gw, 0.0, 1.0, corners.device)
        q = torch.stack([(gu + m) / (1 + 2 * m), (gv + m) / (1 + 2 * m)], -1) * 2 - 1
        y = F.grid_sample(
            y, q[None].expand(N, -1, -1, -1), mode="bilinear", align_corners=True
        )
        base = quad_points(corners, gu, gv).permute(0, 3, 1, 2)
        size = (
            (corners[:, 1] - corners[:, 0]).norm(dim=-1)
            + (corners[:, 2] - corners[:, 3]).norm(dim=-1)
            + (corners[:, 3] - corners[:, 0]).norm(dim=-1)
            + (corners[:, 2] - corners[:, 1]).norm(dim=-1)
        )
        g2 = base + y[:, :2] * (size / 4).view(N, 1, 1, 1)
        return g2, y[:, 2:5], y[:, 5:6]

    def forward(self, image):
        feats = self.features(image)
        heat, off = self.detect(feats[1], feats[2])
        return heat, off, feats


def decode(heat, off, input_hw, thresh=0.3, k=16):
    """Per photo: [(score, corners (4, 2) normalized)] of heatmap peaks above `thresh` (3x3 max), best first, at
    most k."""
    B = heat.shape[0]
    H, W = input_hw
    p = torch.sigmoid(heat.float())
    peak = (p == F.max_pool2d(p, 3, 1, 1)) & (p > thresh)
    out = []
    for b in range(B):
        ys, xs = torch.nonzero(peak[b, 0], as_tuple=True)
        s = p[b, 0, ys, xs]
        order = torch.argsort(s, descending=True)[:k]
        res = []
        for i in order.tolist():
            y, x = int(ys[i]), int(xs[i])
            c = torch.tensor(
                [(x + 0.5) * DET_STRIDE, (y + 0.5) * DET_STRIDE], device=heat.device
            )
            o = off[b, :, y, x].float().view(4, 2) * CORNER_UNIT + c
            res.append((float(s[i]), o / o.new_tensor([W, H]) * 2 - 1))
        out.append(res)
    return out


def grid_corners(g2):
    """(N, 2, gh, gw) grids -> (N, 4, 2) corners [c00, c10, c11, c01]."""
    return torch.stack(
        [g2[:, :, 0, 0], g2[:, :, 0, -1], g2[:, :, -1, -1], g2[:, :, -1, 0]], 1
    )
