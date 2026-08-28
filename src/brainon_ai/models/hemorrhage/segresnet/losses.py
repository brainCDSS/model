from __future__ import annotations

from monai.losses import DiceCELoss


def build_loss(config: dict):
    cfg = config["loss"]
    name = str(cfg["name"]).lower()
    if name != "dice_ce":
        raise ValueError(f"지원하지 않는 loss: {name}. losses.py에 새 loss를 등록하세요.")
    return DiceCELoss(
        include_background=bool(cfg["include_background"]),
        to_onehot_y=True,
        softmax=True,
        batch=bool(cfg.get("batch_dice", False)),
        lambda_dice=float(cfg["dice_weight"]),
        lambda_ce=float(cfg["ce_weight"]),
    )
