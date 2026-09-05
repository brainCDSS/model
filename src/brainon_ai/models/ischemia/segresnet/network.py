"""I-S00 baseline and I-S10 attention-gated SegResNet construction."""

import torch
from monai.networks.nets import SegResNet
from torch import nn


class AdditiveAttentionGate(nn.Module):
    """Additive attention for one SegResNet skip connection."""

    def __init__(self, channels: int, spatial_dims: int = 3):
        super().__init__()

        conv = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        hidden = max(1, channels // 2)

        self.skip_projection = conv(
            channels, hidden, kernel_size=1, bias=True
        )
        self.gate_projection = conv(
            channels, hidden, kernel_size=1, bias=True
        )
        self.score = conv(
            hidden, 1, kernel_size=1, bias=True
        )

    def forward(
        self,
        skip: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:

        attention = torch.sigmoid(
            self.score(
                torch.relu(
                    self.skip_projection(skip)
                    + self.gate_projection(gate)
                )
            )
        )

        return skip * attention


class SegResNetAttentionGate(SegResNet):
    """
    I-S10:
    MONAI SegResNet baseline + Additive Attention Gate
    on every encoder-decoder skip connection.

    Encoder / residual blocks / upsampling / output head are unchanged.
    Only the skip feature is gated before the original residual addition.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        decoder_channels = [
            self.init_filters
            * 2 ** (len(self.blocks_up) - stage - 1)
            for stage in range(len(self.blocks_up))
        ]

        self.attention_gates = nn.ModuleList(
            AdditiveAttentionGate(
                channels=channels,
                spatial_dims=self.spatial_dims,
            )
            for channels in decoder_channels
        )

    def decode(
        self,
        x: torch.Tensor,
        down_x: list[torch.Tensor],
    ) -> torch.Tensor:

        for stage, (upsample, blocks) in enumerate(
            zip(self.up_samples, self.up_layers, strict=True)
        ):
            upsampled = upsample(x)
            skip = down_x[stage + 1]

            attended_skip = self.attention_gates[stage](
                skip=skip,
                gate=upsampled,
            )

            # 기존 SegResNet의 skip fusion은 addition.
            # 차이는 skip을 Attention Gate로 먼저 거른다는 점뿐.
            x = blocks(upsampled + attended_skip)

        return (
            self.conv_final(x)
            if self.use_conv_final
            else x
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, down_x = self.encode(x)
        down_x.reverse()
        return self.decode(x, down_x)


def build_model(config: dict):
    kwargs = {
        "spatial_dims": 3,
        "init_filters": int(config["model"]["init_filters"]),
        "in_channels": 2,
        "out_channels": 1,
        "dropout_prob": 0.0,
    }
    model_name = config["model"]["name"]
    if model_name == "segresnet_early_fusion":
        return SegResNet(**kwargs)
    if model_name == "segresnet_attention_gate":
        return SegResNetAttentionGate(**kwargs)
    raise ValueError(f"지원하지 않는 SegResNet 모델: {model_name!r}")
