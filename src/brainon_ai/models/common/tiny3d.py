from __future__ import annotations

import torch
from torch import nn


class TinySegNet(nn.Module):
    """Small CPU-test network; production adapters can replace it without changing the engine."""

    def __init__(self, in_channels: int = 1, features: int = 4) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv3d(in_channels, features, 3, padding=1),
            nn.InstanceNorm3d(features),
            nn.ReLU(inplace=True),
            nn.Conv3d(features, features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(features, 1, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)
