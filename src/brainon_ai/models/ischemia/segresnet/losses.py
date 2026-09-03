"""Validated I-S00 loss construction."""

from monai.losses import DiceCELoss


def build_loss():
    return DiceCELoss(
        sigmoid=True,
        squared_pred=True,
    )
