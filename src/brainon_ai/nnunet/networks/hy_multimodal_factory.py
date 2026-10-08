"""Network factory for hy DWI/ADC nnU-Net experiments."""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from brainon_ai.nnunet.networks.multimodal import DualEncoderGatedUNet, DualStemUNet


def build_multimodal_network(config, plans_manager, configuration_manager, num_output_channels, enable_deep_supervision):
    name = config["name"]
    def baseline(input_channels):
        return nnUNetTrainer.build_network_architecture(plans_manager, configuration_manager, input_channels, num_output_channels, enable_deep_supervision)
    if name == "dual_stem_resenc":
        return DualStemUNet(baseline(1))
    if name == "dual_encoder_gated_resenc":
        return DualEncoderGatedUNet(baseline(1), baseline(1))
    raise ValueError(f"지원하지 않는 multimodal nnU-Net: {name}")
