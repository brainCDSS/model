from __future__ import annotations

from monai.losses import DiceCELoss


def build_loss(config: dict) -> DiceCELoss:
    loss = config["loss"]

    if str(loss["name"]).lower() != "dice_ce":
        raise ValueError(
            f"Unsupported A-S00 loss: {loss['name']}"
        )

    return DiceCELoss(
        include_background=bool(
            loss["include_background"]
        ),
        to_onehot_y=True,
        softmax=True,
        batch=bool(loss["batch_dice"]),
        lambda_dice=float(loss["dice_weight"]),
        lambda_ce=float(loss["ce_weight"]),
    )
