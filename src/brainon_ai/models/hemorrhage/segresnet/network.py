from __future__ import annotations

from monai.networks.nets import SegResNet


def build_model(config: dict):
    cfg = config["model"]
    return SegResNet(
        spatial_dims=int(cfg["spatial_dims"]),
        in_channels=int(cfg["in_channels"]),
        out_channels=int(cfg["out_channels"]),
        init_filters=int(cfg["init_filters"]),
        blocks_down=tuple(int(x) for x in cfg["blocks_down"]),
        blocks_up=tuple(int(x) for x in cfg["blocks_up"]),
        dropout_prob=float(cfg["dropout_prob"]),
    )

