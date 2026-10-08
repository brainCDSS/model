"""DWI/ADC multimodal wrappers for residual-encoder nnU-Net."""

from copy import deepcopy
import torch
from torch import nn


def _output_channels(encoder):
    channels = getattr(encoder, "output_channels", None)
    if channels is None:
        raise AttributeError("nnU-Net encoder output_channels를 찾을 수 없습니다")
    return [int(channel) for channel in channels]


class DualStemUNet(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.encoder = base.encoder
        self.decoder = base.decoder
        self.dwi_stem = self.encoder.stem
        self.adc_stem = deepcopy(self.encoder.stem)
        channels = _output_channels(self.encoder)[0]
        self.stem_fusion = nn.Conv3d(2 * channels, channels, 1, bias=False)
        self.encoder.stem = None

    def forward(self, image):
        features = self.stem_fusion(torch.cat([self.dwi_stem(image[:, :1]), self.adc_stem(image[:, 1:2])], dim=1))
        skips = []
        for stage in self.encoder.stages:
            features = stage(features)
            skips.append(features)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)


class DualEncoderGatedUNet(nn.Module):
    def __init__(self, dwi_network, adc_network):
        super().__init__()
        self.encoder = dwi_network.encoder
        self.adc_encoder = adc_network.encoder
        self.decoder = dwi_network.decoder
        self.fusion_gates = nn.ModuleList([nn.Conv3d(2 * channels, channels, 1) for channels in _output_channels(self.encoder)])

    def forward(self, image):
        dwi_skips = self.encoder(image[:, :1])
        adc_skips = self.adc_encoder(image[:, 1:2])
        fused = []
        for dwi, adc, gate_layer in zip(dwi_skips, adc_skips, self.fusion_gates, strict=True):
            gate = torch.sigmoid(gate_layer(torch.cat([dwi, adc], dim=1)))
            fused.append(gate * dwi + (1 - gate) * adc)
        return self.decoder(fused)

    def compute_conv_feature_map_size(self, input_size):
        return 2 * self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)
