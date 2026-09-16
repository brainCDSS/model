"""Configurable multimodal SegResNet variants for hy ischemia experiments."""

import torch
from monai.networks.nets import SegResNetDS
from torch import nn


class DualStem(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        def branch():
            return nn.Sequential(nn.Conv3d(1, channels, 3, padding=1, bias=False), nn.InstanceNorm3d(channels, affine=True), nn.ReLU(inplace=True))
        self.dwi = branch()
        self.adc = branch()
        self.fusion = nn.Conv3d(2 * channels, channels, 1, bias=False)

    def forward(self, image):
        return self.fusion(torch.cat([self.dwi(image[:, :1]), self.adc(image[:, 1:2])], dim=1))


class DualEncoderGatedSegResNetDS(nn.Module):
    """Independent DWI/ADC encoders, gated feature fusion at every scale, one decoder."""
    def __init__(self, config):
        super().__init__()
        template = SegResNetDS(**_common_kwargs(config, in_channels=1))
        adc_template = SegResNetDS(**_common_kwargs(config, in_channels=1))
        self.dwi_encoder = template.encoder
        self.adc_encoder = adc_template.encoder
        self.up_layers = template.up_layers
        self.dsdepth = int(config["model"]["deep_supervision_depth"])
        init_filters = int(config["model"]["init_filters"])
        channels = [init_filters * (2 ** level) for level in range(len(self.dwi_encoder.layers))]
        self.gates = nn.ModuleList([nn.Conv3d(2 * channel, channel, 1) for channel in channels])

    def forward(self, image):
        dwi_features = self.dwi_encoder(image[:, :1])
        adc_features = self.adc_encoder(image[:, 1:2])
        fused = []
        for dwi, adc, gate_layer in zip(dwi_features, adc_features, self.gates, strict=True):
            gate = torch.sigmoid(gate_layer(torch.cat([dwi, adc], dim=1)))
            fused.append(gate * dwi + (1.0 - gate) * adc)
        fused.reverse()
        features = fused.pop(0)
        outputs = []
        for index, level in enumerate(self.up_layers):
            features = level["upsample"](features)
            features = level["blocks"](features + fused.pop(0))
            if len(self.up_layers) - index <= self.dsdepth:
                outputs.append(level["head"](features))
        outputs.reverse()
        return outputs if self.training and len(outputs) > 1 else outputs[0]


def _common_kwargs(config, in_channels=None, preprocess=None):
    model = config["model"]
    return {
        "spatial_dims": 3,
        "init_filters": int(model["init_filters"]),
        "in_channels": int(in_channels or config["data"]["in_channels"]),
        "out_channels": int(model["out_channels"]),
        "dsdepth": int(model.get("deep_supervision_depth", 1)),
        "preprocess": preprocess,
        "resolution": tuple(config["preprocessing"]["spacing"]),
    }


def build_model(config: dict):
    name = config["model"]["name"]
    if name == "dual_encoder_gated_segresnet_ds":
        return DualEncoderGatedSegResNetDS(config)
    if name == "dual_stem_segresnet_ds":
        channels = int(config["model"]["init_filters"])
        stem = DualStem(channels)
        return SegResNetDS(**_common_kwargs(config, in_channels=channels, preprocess=stem))
    if name in {"segresnet_ds", "segresnet_ds_dice_focal"}:
        return SegResNetDS(**_common_kwargs(config))
    raise ValueError(f"지원하지 않는 모델: {name!r}")
