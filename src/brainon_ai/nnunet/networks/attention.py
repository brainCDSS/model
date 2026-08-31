from __future__ import annotations

import numpy as np
import torch
from torch import nn


class AttentionGate(nn.Module):
    """Additive attention gate for a U-Net skip tensor."""

    def __init__(self, channels: int, conv_op: type[nn.Module]):
        super().__init__()
        hidden = max(1, channels // 2)
        self.skip_projection = conv_op(channels, hidden, kernel_size=1, bias=True)
        self.gate_projection = conv_op(channels, hidden, kernel_size=1, bias=True)
        self.score = conv_op(hidden, 1, kernel_size=1, bias=True)
        self.activation = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        weights = self.sigmoid(
            self.score(self.activation(self.skip_projection(skip) + self.gate_projection(gate)))
        )
        return skip * weights


class AttentionUNetDecoder(nn.Module):
    """Drop-in wrapper adding attention gates to an nnU-Net decoder."""

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.deep_supervision = decoder.deep_supervision
        self.encoder = decoder.encoder
        self.num_classes = decoder.num_classes
        self.stages = decoder.stages
        self.transpconvs = decoder.transpconvs
        self.seg_layers = decoder.seg_layers
        skip_channels = [
            self.encoder.output_channels[-(stage + 2)] for stage in range(len(self.stages))
        ]
        self.attention_gates = nn.ModuleList(
            AttentionGate(channels, self.encoder.conv_op) for channels in skip_channels
        )

    def forward(self, skips: list[torch.Tensor]):
        low_resolution = skips[-1]
        segmentation_outputs = []
        for stage, (upsample, blocks, attention) in enumerate(
            zip(self.transpconvs, self.stages, self.attention_gates, strict=True)
        ):
            upsampled = upsample(low_resolution)
            skip = attention(skips[-(stage + 2)], upsampled)
            decoded = blocks(torch.cat((upsampled, skip), dim=1))
            if self.deep_supervision:
                segmentation_outputs.append(self.seg_layers[stage](decoded))
            elif stage == len(self.stages) - 1:
                segmentation_outputs.append(self.seg_layers[-1](decoded))
            low_resolution = decoded
        segmentation_outputs.reverse()
        return segmentation_outputs if self.deep_supervision else segmentation_outputs[0]

    def compute_conv_feature_map_size(self, input_size):
        skip_sizes = []
        current_size = list(input_size)
        for stride in self.encoder.strides[:-1]:
            current_size = [size // step for size, step in zip(current_size, stride, strict=True)]
            skip_sizes.append(current_size)

        output = np.int64(0)
        for stage, skip_size in enumerate(reversed(skip_sizes)):
            channels = self.encoder.output_channels[-(stage + 2)]
            hidden = max(1, channels // 2)
            voxels = np.prod(skip_size, dtype=np.int64)
            output += self.stages[stage].compute_conv_feature_map_size(skip_size)
            output += channels * voxels
            output += (2 * hidden + 1) * voxels
            if self.deep_supervision or stage == len(self.stages) - 1:
                output += self.num_classes * voxels
        return output
