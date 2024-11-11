import torch
import torch.nn as nn

from .hybrid_encoder import HybridEncoder
from .presnet import PResNet
from .rtdetrv2_decoder import RTDETRTransformerv2


class RTDETR(nn.Module):
    __inject__ = [
        "backbone",
        "encoder",
        "decoder",
    ]

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = PResNet(**cfg.PResNet)
        self.encoder = HybridEncoder(**cfg.HybridEncoder)
        self.decoder = RTDETRTransformerv2(**cfg.RTDETRTransformerv2)

    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)

        return x

    def deploy(
        self,
    ):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self


# model = RTDETR({})
# weights = torch.load("weights/layout_20241111.pth", map_location="cpu")
# model.load_state_dict(weights["model"])
# print("RTDETR model loaded successfully!")
