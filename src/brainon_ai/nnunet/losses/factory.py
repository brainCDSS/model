from __future__ import annotations

import numpy as np

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss


def build_loss(trainer, config: dict):
    """설정에 등록된 loss를 만들고 nnU-Net deep supervision 계약을 적용합니다."""
    name = str(config.get("name", "dice_ce")).lower()
    if name != "dice_ce":
        raise ValueError(
            f"지원하지 않는 nnU-Net loss: {name}. "
            "새 loss는 ich_nnunet/losses/factory.py에 등록하세요."
        )
    loss = DC_and_CE_loss(
        {
            "batch_dice": trainer.configuration_manager.batch_dice,
            "smooth": float(config.get("smooth", 1e-5)),
            "do_bg": bool(config.get("include_background", False)),
            "ddp": trainer.is_ddp,
        },
        {},
        weight_ce=float(config.get("ce_weight", 1.0)),
        weight_dice=float(config.get("dice_weight", 1.0)),
        ignore_label=trainer.label_manager.ignore_label,
        dice_class=MemoryEfficientSoftDiceLoss,
    )
    if trainer.enable_deep_supervision:
        scales = trainer._get_deep_supervision_scales()
        weights = np.asarray([1 / (2**i) for i in range(len(scales))], dtype=np.float64)
        weights[-1] = 1e-6 if trainer.is_ddp and not trainer._do_i_compile() else 0
        weights /= weights.sum()
        loss = DeepSupervisionWrapper(loss, weights)
    return loss

