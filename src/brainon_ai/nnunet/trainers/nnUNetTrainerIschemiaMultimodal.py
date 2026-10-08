"""Trainer for I-N110/I-N120 multimodal ischemia experiments."""

from brainon_ai.nnunet.config import load_config
from brainon_ai.nnunet.networks.hy_multimodal_factory import build_multimodal_network
from brainon_ai.nnunet.trainers.configurable_trainer import nnUNetTrainerICHBaseline


class nnUNetTrainerIschemiaMultimodal(nnUNetTrainerICHBaseline):
    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager, num_input_channels, num_output_channels, enable_deep_supervision=True):
        if num_input_channels != 2:
            raise ValueError(f"DWI+ADC 2채널이 필요합니다: {num_input_channels}")
        return build_multimodal_network(load_config()["network"], plans_manager, configuration_manager, num_output_channels, enable_deep_supervision)
