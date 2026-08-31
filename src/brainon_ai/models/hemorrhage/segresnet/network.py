from __future__ import annotations

import torch
from monai.networks.nets import SegResNet
from torch import nn


def _conv(spatial_dims: int):
    return nn.Conv3d if spatial_dims == 3 else nn.Conv2d


class AttentionGate(nn.Module):
    """Additive attention for a SegResNet skip connection."""

    def __init__(self, channels: int, spatial_dims: int):
        super().__init__()
        conv = _conv(spatial_dims)
        hidden = max(1, channels // 2)
        self.skip_projection = conv(channels, hidden, kernel_size=1)
        self.gate_projection = conv(channels, hidden, kernel_size=1)
        self.score = conv(hidden, 1, kernel_size=1)

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        weights = torch.sigmoid(
            self.score(torch.relu(self.skip_projection(skip) + self.gate_projection(gate)))
        )
        return skip * weights


class MultiScaleContext(nn.Module):
    """Residual multi-scale dilated context block at the bottleneck."""

    def __init__(self, channels: int, spatial_dims: int, dilations=(1, 2, 4)):
        super().__init__()
        conv = _conv(spatial_dims)
        self.branches = nn.ModuleList(
            [conv(channels, channels, kernel_size=1)]
            + [
                conv(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
                for dilation in dilations
            ]
        )
        self.fuse = nn.Sequential(
            conv(channels * len(self.branches), channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + self.fuse(torch.cat([branch(tensor) for branch in self.branches], dim=1))


class SegResNetVariant(SegResNet):
    def __init__(self, *, attention: bool, multiscale: bool, context_dilations=(1, 2, 4), **kwargs):
        super().__init__(**kwargs)
        decoder_channels = [
            self.init_filters * 2 ** (len(self.blocks_up) - stage - 1)
            for stage in range(len(self.blocks_up))
        ]
        self.attention_gates = (
            nn.ModuleList(AttentionGate(channels, self.spatial_dims) for channels in decoder_channels)
            if attention
            else None
        )
        bottleneck_channels = self.init_filters * 2 ** (len(self.blocks_down) - 1)
        self.context = (
            MultiScaleContext(bottleneck_channels, self.spatial_dims, context_dilations)
            if multiscale
            else nn.Identity()
        )

    def decode(self, x: torch.Tensor, down_x: list[torch.Tensor]) -> torch.Tensor:
        for stage, (upsample, blocks) in enumerate(zip(self.up_samples, self.up_layers, strict=True)):
            upsampled = upsample(x)
            skip = down_x[stage + 1]
            if self.attention_gates is not None:
                skip = self.attention_gates[stage](skip, upsampled)
            x = blocks(upsampled + skip)
        return self.conv_final(x) if self.use_conv_final else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, down_x = self.encode(x)
        x = self.context(x)
        down_x.reverse()
        return self.decode(x, down_x)


def build_model(config: dict):
    cfg = config["model"]
    name = str(cfg.get("name", "segresnet")).lower()
    common = {
        "spatial_dims": int(cfg["spatial_dims"]),
        "in_channels": int(cfg["in_channels"]),
        "out_channels": int(cfg["out_channels"]),
        "init_filters": int(cfg["init_filters"]),
        "blocks_down": tuple(int(x) for x in cfg["blocks_down"]),
        "blocks_up": tuple(int(x) for x in cfg["blocks_up"]),
        "dropout_prob": float(cfg["dropout_prob"]),
    }
    if name == "segresnet":
        return SegResNet(**common)
    if name == "segresnet_attention_gate":
        return SegResNetVariant(attention=True, multiscale=False, **common)
    if name == "segresnet_multiscale_context":
        dilations = tuple(int(value) for value in cfg.get("context_dilations", (1, 2, 4)))
        return SegResNetVariant(
            attention=False, multiscale=True, context_dilations=dilations, **common
        )
    raise ValueError(f"지원하지 않는 SegResNet model.name: {name}")
