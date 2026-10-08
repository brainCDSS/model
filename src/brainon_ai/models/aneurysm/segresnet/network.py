from __future__ import annotations

from monai.networks.nets import SegResNet


def build_model(config: dict) -> SegResNet:
    """Build the unmodified MONAI SegResNet for A-S00."""

    model = config["model"]

    if str(model["name"]).lower() != "segresnet":
        raise ValueError(
            "A-S00 permits only the standard MONAI SegResNet"
        )

    return SegResNet(
        spatial_dims=int(model["spatial_dims"]),
        in_channels=int(model["in_channels"]),
        out_channels=int(model["out_channels"]),
        init_filters=int(model["init_filters"]),
        blocks_down=tuple(
            int(value) for value in model["blocks_down"]
        ),
        blocks_up=tuple(
            int(value) for value in model["blocks_up"]
        ),
        dropout_prob=float(model["dropout_prob"]),
    )
