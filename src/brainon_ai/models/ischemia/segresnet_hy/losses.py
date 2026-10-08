"""Configurable deep-supervision losses for hy SegResNet experiments."""

import torch
from monai.losses import DiceCELoss, DiceFocalLoss, GeneralizedDiceFocalLoss, TverskyLoss


class DeepSupervisionLoss(torch.nn.Module):
    def __init__(self, loss, weights):
        super().__init__()
        self.loss = loss
        self.weights = [float(weight) for weight in weights]

    def forward(self, outputs, target):
        if not isinstance(outputs, (list, tuple)):
            return self.loss(outputs, target)
        losses = []
        for output, weight in zip(outputs, self.weights, strict=False):
            resized = torch.nn.functional.interpolate(target, size=output.shape[2:], mode="nearest")
            losses.append(weight * self.loss(output, resized))
        return sum(losses) / sum(self.weights[:len(losses)])


def build_loss(config):
    cfg = config["loss"]
    name = cfg["name"]
    if name == "dice_focal":
        loss = DiceFocalLoss(sigmoid=True, squared_pred=True, gamma=float(cfg.get("gamma", 2.0)))
    elif name == "dice_ce":
        loss = DiceCELoss(sigmoid=True, squared_pred=True)
    elif name == "tversky":
        loss = TverskyLoss(sigmoid=True, alpha=float(cfg.get("alpha", 0.3)), beta=float(cfg.get("beta", 0.7)))
    elif name == "generalized_dice_focal":
        loss = GeneralizedDiceFocalLoss(sigmoid=True, gamma=float(cfg.get("gamma", 2.0)))
    else:
        raise ValueError(f"지원하지 않는 loss: {name}")
    return DeepSupervisionLoss(loss, cfg.get("deep_supervision_weights", [1.0]))
