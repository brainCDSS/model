from __future__ import annotations

from brainon_ai.nnunet.networks.attention_unet import (
    attach_attention_decoder,
)
from brainon_ai.nnunet.trainers.nnUNetTrainerIschemiaBaseline import (
    nnUNetTrainerIschemiaBaseline,
)


class nnUNetTrainerIschemiaAttention(
    nnUNetTrainerIschemiaBaseline
):
    """I-N10: I-N00 baseline + Attention Gate on all U-Net skips."""

    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        network = (
            nnUNetTrainerIschemiaBaseline
            .build_network_architecture(
                plans_manager,
                configuration_manager,
                num_input_channels,
                num_output_channels,
                enable_deep_supervision,
            )
        )

        return attach_attention_decoder(
            network=network,
            configuration_manager=configuration_manager,
            num_output_channels=num_output_channels,
            enable_deep_supervision=enable_deep_supervision,
        )
