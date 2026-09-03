"""Validated I-S00 MONAI SegResNet construction."""

from monai.networks.nets import SegResNet


def build_model(config: dict):
    return SegResNet(
        spatial_dims=3,
        init_filters=int(config["model"]["init_filters"]),
        in_channels=2,
        out_channels=1,
        dropout_prob=0.0,
    )
