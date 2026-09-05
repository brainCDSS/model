from __future__ import annotations

import torch
from torch import nn

from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder


class AdditiveAttentionGate(nn.Module):
    """Additive Attention Gate for one U-Net skip connection."""

    def __init__(
        self,
        conv_op,
        skip_channels: int,
        gate_channels: int,
        inter_channels: int | None = None,
    ) -> None:
        super().__init__()

        if inter_channels is None:
            inter_channels = max(1, skip_channels // 2)

        self.theta_x = conv_op(
            skip_channels,
            inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.phi_g = conv_op(
            gate_channels,
            inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.relu = nn.ReLU(inplace=True)
        self.psi = conv_op(
            inter_channels,
            1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x_skip: torch.Tensor,
        g: torch.Tensor,
    ) -> torch.Tensor:
        attention = self.relu(
            self.theta_x(x_skip) + self.phi_g(g)
        )
        attention = self.sigmoid(
            self.psi(attention)
        )
        return x_skip * attention


class AttentionUNetDecoder(UNetDecoder):
    """nnU-Net default decoder with an Attention Gate on every skip."""

    def __init__(
        self,
        encoder,
        num_classes: int,
        n_conv_per_stage,
        deep_supervision: bool,
    ) -> None:
        super().__init__(
            encoder=encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage,
            deep_supervision=deep_supervision,
        )

        n_stages_encoder = len(encoder.output_channels)
        gates = []

        for s in range(1, n_stages_encoder):
            skip_channels = encoder.output_channels[-(s + 1)]

            gates.append(
                AdditiveAttentionGate(
                    conv_op=encoder.conv_op,
                    skip_channels=skip_channels,
                    gate_channels=skip_channels,
                    inter_channels=max(1, skip_channels // 2),
                )
            )

        self.attention_gates = nn.ModuleList(gates)

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []

        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            skip = skips[-(s + 2)]

            attended_skip = self.attention_gates[s](
                x_skip=skip,
                g=x,
            )

            x = torch.cat(
                (x, attended_skip),
                dim=1,
            )
            x = self.stages[s](x)

            if self.deep_supervision:
                seg_outputs.append(
                    self.seg_layers[s](x)
                )
            elif s == (len(self.stages) - 1):
                seg_outputs.append(
                    self.seg_layers[-1](x)
                )

            lres_input = x

        seg_outputs = seg_outputs[::-1]

        if self.deep_supervision:
            return seg_outputs
        return seg_outputs[0]


def attach_attention_decoder(
    network: nn.Module,
    configuration_manager,
    num_output_channels: int,
    enable_deep_supervision: bool,
) -> nn.Module:
    """Replace only the default nnU-Net decoder with AttentionUNetDecoder."""

    if not hasattr(network, "encoder") or not hasattr(network, "decoder"):
        raise TypeError(
            "I-N10 expects the Dataset502 3d_fullres PlainConvUNet."
        )

    init_kwargs = configuration_manager.network_arch_init_kwargs
    n_conv_per_stage_decoder = init_kwargs.get(
        "n_conv_per_stage_decoder"
    )

    if n_conv_per_stage_decoder is None:
        raise KeyError(
            "n_conv_per_stage_decoder was not found in nnUNetPlans."
        )

    decoder = AttentionUNetDecoder(
        encoder=network.encoder,
        num_classes=num_output_channels,
        n_conv_per_stage=n_conv_per_stage_decoder,
        deep_supervision=enable_deep_supervision,
    )

    initializer = getattr(network, "initialize", None)
    if initializer is not None:
        decoder.apply(initializer)

    network.decoder = decoder
    return network
