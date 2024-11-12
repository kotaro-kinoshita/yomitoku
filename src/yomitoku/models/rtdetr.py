import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from .layers.rtdetr_backbone import PResNet
from .layers.rtdetr_hybrid_encoder import HybridEncoder
from .layers.rtdetrv2_decoder import RTDETRTransformerv2


class RTDETRv2(nn.Module, PyTorchModelHubMixin):
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
