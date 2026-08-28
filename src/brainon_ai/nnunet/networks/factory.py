from __future__ import annotations

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


def build_network(
    config: dict,
    plans_manager,
    configuration_manager,
    num_input_channels: int,
    num_output_channels: int,
    enable_deep_supervision: bool,
):
    """네트워크 registry. baseline은 nnU-Net plans의 기본 구조를 그대로 사용합니다."""
    name = str(config.get("name", "default")).lower()
    if name == "default":
        return nnUNetTrainer.build_network_architecture(
            plans_manager,
            configuration_manager,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        )
    raise ValueError(
        f"지원하지 않는 network: {name}. "
        "새 구조는 ich_nnunet/networks/factory.py에 등록하세요."
    )

