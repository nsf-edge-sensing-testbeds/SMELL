from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights

from .se import SqueezeExcitation


class ResNeXtSEEncoder(nn.Module):
    """ResNeXt-50 (32x4d) + post-backbone squeeze-excitation.

    The paper describes ImageNet-21K initialization. torchvision does not ship
    that exact checkpoint, so `pretrained=True` uses torchvision's DEFAULT
    ImageNet weights as a practical substitute unless an external checkpoint
    is loaded by the caller.
    """

    def __init__(self, out_dim: int = 512, pretrained: bool = False):
        super().__init__()
        weights = ResNeXt50_32X4D_Weights.DEFAULT if pretrained else None
        net = resnext50_32x4d(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.se = SqueezeExcitation(2048)
        self.region_proj = nn.Linear(2048, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.se(x)
        # Keep spatial regions so cross-attention can align regions and tokens.
        x = x.flatten(2).transpose(1, 2)  # [B, R, 2048]
        return self.region_proj(x)        # [B, R, D]
