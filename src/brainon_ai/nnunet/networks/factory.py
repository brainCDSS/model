from __future__ import annotations

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from brainon_ai.nnunet.networks.attention import AttentionUNetDecoder


def build_network(
    config: dict,
    plans_manager,
    configuration_manager,
    num_input_channels: int,
    num_output_channels: int,
    enable_deep_supervision: bool,
):
    """Network registry for the baseline and hemorrhage ablations."""
    name = str(config.get("name", "default")).lower()
    if name in {"default", "resenc_m", "resenc_m_attention"}:
        network = nnUNetTrainer.build_network_architecture(
            plans_manager,
            configuration_manager,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        )
        if name == "resenc_m_attention":
            if not hasattr(network, "decoder"):
                raise TypeError("Attention Gate를 적용할 nnU-Net decoder가 없습니다.")
            network.decoder = AttentionUNetDecoder(network.decoder)
            if hasattr(network, "initialize"):
                network.decoder.attention_gates.apply(network.initialize)
        return network
    raise ValueError(
        f"지원하지 않는 network: {name}. "
        "새 구조는 brainon_ai/nnunet/networks/factory.py에 등록하세요."
    )
