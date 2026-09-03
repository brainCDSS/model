from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PresenceAwareDiceCELoss(nn.Module):
    """Dice over GT-present classes plus background-weighted CE."""

    def __init__(
        self,
        *,
        num_classes: int,
        background_ce_weight: float,
        dice_weight: float,
        ce_weight: float,
        smooth: float = 1e-5,
    ):
        super().__init__()

        if num_classes != 53:
            raise ValueError(
                "A-S00 requires exactly 53 classes"
            )

        if not 0.0 < background_ce_weight <= 1.0:
            raise ValueError(
                "background_ce_weight must be in (0, 1]"
            )

        self.num_classes = int(num_classes)
        self.background_ce_weight = float(
            background_ce_weight
        )
        self.dice_weight = float(dice_weight)
        self.ce_weight = float(ce_weight)
        self.smooth = float(smooth)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if target.ndim == logits.ndim:
            if target.shape[1] != 1:
                raise ValueError(
                    "Target channel dimension must be 1"
                )
            target = target[:, 0]

        target = target.long()

        if logits.shape[1] != self.num_classes:
            raise ValueError(
                "A-S00 logits must contain 53 channels"
            )

        if target.numel():
            if (
                int(target.detach().min()) < 0
                or int(target.detach().max())
                >= self.num_classes
            ):
                raise ValueError(
                    "Target contains an invalid class"
                )

        probabilities = torch.softmax(
            logits.float(),
            dim=1,
        )

        one_hot = F.one_hot(
            target,
            num_classes=self.num_classes,
        ).movedim(-1, 1).to(
            dtype=probabilities.dtype
        )

        predicted_fg = probabilities[:, 1:]
        target_fg = one_hot[:, 1:]

        spatial_axes = tuple(
            range(2, logits.ndim)
        )

        intersection = (
            predicted_fg * target_fg
        ).sum(dim=spatial_axes)

        predicted_volume = predicted_fg.sum(
            dim=spatial_axes
        )
        target_volume = target_fg.sum(
            dim=spatial_axes
        )

        present = target_volume > 0

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            predicted_volume
            + target_volume
            + self.smooth
        )

        if torch.any(present):
            dice_loss = (
                1.0 - dice
            )[present].mean()
        else:
            dice_loss = probabilities.sum() * 0.0

        class_weights = torch.ones(
            self.num_classes,
            dtype=torch.float32,
            device=logits.device,
        )
        class_weights[0] = self.background_ce_weight

        ce_loss = F.cross_entropy(
            logits.float(),
            target,
            weight=class_weights,
        )

        return (
            self.dice_weight * dice_loss
            + self.ce_weight * ce_loss
        )


def build_loss(config: dict) -> PresenceAwareDiceCELoss:
    loss = config["loss"]

    if (
        str(loss["name"]).lower()
        != "presence_aware_dice_ce"
    ):
        raise ValueError(
            f"Unsupported A-S00 loss: {loss['name']}"
        )

    return PresenceAwareDiceCELoss(
        num_classes=int(
            config["model"]["out_channels"]
        ),
        background_ce_weight=float(
            loss["background_ce_weight"]
        ),
        dice_weight=float(loss["dice_weight"]),
        ce_weight=float(loss["ce_weight"]),
    )
