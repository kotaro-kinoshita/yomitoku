from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models._utils import IntermediateLayerGetter

from .layers.dbnet_feature_attention import ScaleFeatureSelection


class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        return_layers = {
            "layer1": "layer1",
            "layer2": "layer2",
            "layer3": "layer3",
            "layer4": "layer4",
        }

        self.body = IntermediateLayerGetter(
            backbone, return_layers=return_layers
        )

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name="resnet50", dilation=True):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=False,
        )
        super().__init__(backbone)


class DBNetDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_dim,
        adaptive=False,
        serial=False,
        smooth=False,
        k=50,
    ):
        super().__init__()
        self.d_model = hidden_dim
        self.n_layers = len(in_channels)
        self.k = k
        self.training = True
        self.input_proj = nn.ModuleDict(
            {
                "layer1": nn.Conv2d(
                    in_channels[0], self.d_model, 1, bias=False
                ),
                "layer2": nn.Conv2d(
                    in_channels[1], self.d_model, 1, bias=False
                ),
                "layer3": nn.Conv2d(
                    in_channels[2], self.d_model, 1, bias=False
                ),
                "layer4": nn.Conv2d(
                    in_channels[3], self.d_model, 1, bias=False
                ),
            }
        )

        self.upsample_2x = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

        self.out_proj = nn.ModuleDict(
            {
                "layer1": nn.Conv2d(
                    self.d_model, self.d_model // 4, 3, padding=1, bias=False
                ),
                "layer2": nn.Sequential(
                    nn.Conv2d(
                        self.d_model,
                        self.d_model // 4,
                        3,
                        padding=1,
                        bias=False,
                    ),
                    nn.Upsample(
                        scale_factor=2, mode="bilinear", align_corners=False
                    ),
                ),
                "layer3": nn.Sequential(
                    nn.Conv2d(
                        self.d_model,
                        self.d_model // 4,
                        3,
                        padding=1,
                        bias=False,
                    ),
                    nn.Upsample(
                        scale_factor=4, mode="bilinear", align_corners=False
                    ),
                ),
                "layer4": nn.Sequential(
                    nn.Conv2d(
                        self.d_model,
                        self.d_model // 4,
                        3,
                        padding=1,
                        bias=False,
                    ),
                    nn.Upsample(
                        scale_factor=4, mode="bilinear", align_corners=False
                    ),
                ),
            }
        )

        self.binarize = nn.Sequential(
            nn.Conv2d(
                self.d_model, self.d_model // 4, 3, padding=1, bias=False
            ),
            nn.BatchNorm2d(self.d_model // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.d_model // 4, self.d_model // 4, 2, 2),
            nn.BatchNorm2d(self.d_model // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.d_model // 4, 1, 2, 2),
            nn.Sigmoid(),
        )

        self.adaptive = adaptive
        self.serial = serial
        if self.adaptive:
            self.thresh = self._init_thresh(
                self.d_model,
                serial=serial,
                smooth=smooth,
                bias=False,
            )
            self.thresh.apply(self.weights_init)

        self.binarize.apply(self.weights_init)

        for layer in self.input_proj.values():
            layer.apply(self.weights_init)

        for layer in self.out_proj.values():
            layer.apply(self.weights_init)

        self.concat_attention = ScaleFeatureSelection(
            self.d_model,
            self.d_model // 4,
            attention_type="scale_channel_spatial",
        )

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(1e-4)

    def _init_thresh(
        self, inner_channels, serial=False, smooth=False, bias=False
    ):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(
            nn.Conv2d(
                in_channels, inner_channels // 4, 3, padding=1, bias=bias
            ),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            self._init_upsample(
                inner_channels // 4,
                inner_channels // 4,
                smooth=smooth,
                bias=bias,
            ),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            self._init_upsample(
                inner_channels // 4, 1, smooth=smooth, bias=bias
            ),
            nn.Sigmoid(),
        )
        return self.thresh

    def _init_upsample(
        self, in_channels, out_channels, smooth=False, bias=False
    ):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias),
            ]
            if out_channels == 1:
                module_list.append(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False,
                    )
                )

            return nn.Sequential(*module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))

    def forward(self, features):
        for layer, feature in features.items():
            features[layer] = self.input_proj[layer](feature)

        layers = ["layer4", "layer3", "layer2", "layer1"]
        for i in range(self.n_layers - 1):
            feature_bottom = features[layers[i]]
            feature_top = features[layers[i + 1]]

            bh, bw = feature_bottom.shape[-2:]
            th, tw = feature_top.shape[-2:]

            if bh != th or bw != tw:
                feature_bottom = F.interpolate(
                    feature_bottom,
                    size=(th, tw),
                    mode="bilinear",
                    align_corners=False,
                )

            features[layers[i + 1]] = feature_bottom + feature_top

        fp = []
        for layer, feature in features.items():
            fp.append(self.out_proj[layer](feature))
        fuse = torch.cat(fp[::-1], dim=1)
        fuse = self.concat_attention(fuse, fp[::-1])

        binary = self.binarize(fuse)
        result = OrderedDict(binary=binary)
        return result


class DBNetPlus(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = Backbone(**cfg.backbone)
        self.decoder = DBNetDecoder(**cfg.decoder)

    def forward(self, tensor):
        features = self.backbone(tensor)
        xs = self.decoder(features)
        return xs
