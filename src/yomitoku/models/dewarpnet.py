"""Geometric correction (dewarping) network, ported from docwarp-synth (`docwarp_synth/dewarp/model.py`); module
names follow UVDoc so that the trained state dict loads as is."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin


def _conv(cin: int, cout: int, k: int, stride: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        cin, cout, k, stride, padding=dilation * (k // 2), dilation=dilation
    )


def _conv_bn_act(cin: int, cout: int, dilation: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, 1, padding=dilation, dilation=dilation, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class Block(nn.Module):
    """Residual block; the first block of a layer uses plain (strided) convs, the others dilation-3 convs."""

    def __init__(self, cin: int, cout: int, k: int, stride: int = 1, top: bool = False):
        super().__init__()
        if stride != 1 or top:
            self.conv1, self.conv2 = _conv(cin, cout, k, stride), _conv(cout, cout, k)
        else:
            self.conv1, self.conv2 = (
                nn.Sequential(_conv(cin, cout, k, 1, 3)),
                nn.Sequential(_conv(cout, cout, k, 1, 3)),
            )
        self.bn1, self.bn2 = nn.BatchNorm2d(cout), nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = (
            nn.Sequential(_conv(cin, cout, k, stride), nn.BatchNorm2d(cout))
            if stride != 1 or cin != cout
            else None
        )

    def forward(self, x):
        r = x if self.downsample is None else self.downsample(x)
        y = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(y)) + r)


class Encoder(nn.Module):
    def __init__(self, width: int, k: int, blocks=(3, 4, 6), strides=(1, 2, 2)):
        super().__init__()
        cin = width
        for i, (n, s) in enumerate(zip(blocks, strides)):
            cout = width * 2**i
            setattr(
                self,
                f"layer{i + 1}",
                nn.Sequential(
                    Block(cin, cout, k, s, top=True),
                    *[Block(cout, cout, k) for _ in range(n - 1)],
                ),
            )
            cin = cout
        self.n = len(blocks)

    def forward(self, x):
        for i in range(self.n):
            x = getattr(self, f"layer{i + 1}")(x)
        return x


class SDFDecoder(nn.Module):
    def __init__(self, c16: int, c8: int, c4: int, w: int = 64):
        super().__init__()
        self.up8 = nn.Sequential(_conv_bn_act(c16 + c8, w, 1), _conv_bn_act(w, w, 1))
        self.up4 = nn.Sequential(
            _conv_bn_act(w + c4, w // 2, 1), _conv_bn_act(w // 2, w // 2, 1)
        )
        self.out = nn.Conv2d(w // 2, 1, 1)

    def forward(self, f16, f8, f4):
        x = self.up8(
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
        x = self.up4(
            torch.cat(
                [
                    F.interpolate(
                        x, size=f4.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    f4,
                ],
                1,
            )
        )
        return self.out(x)


class DewarpNet(nn.Module, PyTorchModelHubMixin):
    """Grid regression network for geometric correction (UVDoc architecture: strided head, ResNet with dilated
    residual blocks, six parallel multi-dilation bridges, 2D and 3D grid heads) plus a paper signed-distance head at
    stride 4. Input (B, 3, H, W) RGB in [0, 1]; outputs grid2d (B, 2, 45, 31) -- photo position (x, y) in [-1, 1] of
    a regular flat-page grid --, grid3d (B, 3, 45, 31) and, with cfg.sdf, the paper SDF (B, 1, H/4, W/4)."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        width, k = cfg.width, cfg.kernel
        self.resnet_head = nn.Sequential(
            nn.Conv2d(3, width, k, 2, k // 2, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, k, 2, k // 2, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.resnet_down = Encoder(width, k)
        c = width * 4
        for i, ds in enumerate([(1,), (2,), (5,), (8, 3, 2), (12, 7, 4), (18, 12, 6)]):
            setattr(
                self,
                f"bridge_{i + 1}",
                nn.Sequential(*[_conv_bn_act(c, c, d) for d in ds]),
            )
        self.bridge_concat = nn.Sequential(
            nn.Conv2d(c * 6, c, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True)
        )

        def head(n):
            return nn.Sequential(
                nn.Conv2d(c, width, k, 1, k // 2, bias=False, padding_mode="reflect"),
                nn.BatchNorm2d(width),
                nn.PReLU(),
                nn.Conv2d(width, n, k, 1, k // 2, padding_mode="reflect"),
            )

        self.out_point_positions2D = head(2)
        self.out_point_positions3D = head(3)
        self.sdf = cfg.sdf
        if self.sdf:
            self.sdf_head = SDFDecoder(c, width * 2, width)

    def forward(self, x):
        e = self.resnet_down
        f4 = e.layer1(self.resnet_head(x))
        f8 = e.layer2(f4)
        f = e.layer3(f8)
        f = self.bridge_concat(
            torch.cat([getattr(self, f"bridge_{i}")(f) for i in range(1, 7)], 1)
        )
        out = (self.out_point_positions2D(f), self.out_point_positions3D(f))
        return out + (self.sdf_head(f, f8, f4),) if self.sdf else out
