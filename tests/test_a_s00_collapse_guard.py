from pathlib import Path

import torch

from brainon_ai.common.config import load_config as load_yaml
from brainon_ai.models.aneurysm.segresnet.config import (
    _validate_config,
)
from brainon_ai.models.aneurysm.segresnet.data import (
    deterministic_foreground_unit,
)
from brainon_ai.models.aneurysm.segresnet.losses import (
    build_loss,
)


CONFIG_PATH = Path(
    "configs/experiments/aneurysm/A-S00.yaml"
)


def config():
    value = load_yaml(CONFIG_PATH)
    _validate_config(value)
    return value


def test_attempt_04_contract():
    value = config()

    assert value["experiment_id"] == "A-S00"
    assert value["data"]["patch_size"] == [112, 160, 64]
    assert value["data"]["batch_size"] == 1
    assert (
        value["data"]["foreground_oversample_fraction"]
        == 0.5
    )
    assert (
        value["data"]["sampling_strategy"]
        == "deterministic_global_foreground"
    )
    assert (
        value["validation"]["online_patch"][
            "foreground_fraction"
        ]
        == 0.5
    )
    assert (
        value["loss"]["name"]
        == "presence_aware_dice_ce"
    )
    assert value["loss"]["background_ce_weight"] == 0.001
    assert value["loss"]["dice_weight"] == 1.0
    assert value["loss"]["ce_weight"] == 0.5
    assert value["early_stopping"]["patience"] == 80


def test_exact_forced_foreground_counts():
    train_flags = [
        deterministic_foreground_unit(
            iteration=index,
            total_iterations=250,
            fraction=0.5,
        )
        for index in range(250)
    ]

    validation_flags = [
        deterministic_foreground_unit(
            iteration=index,
            total_iterations=50,
            fraction=0.5,
        )
        for index in range(50)
    ]

    assert sum(train_flags) == 125
    assert sum(validation_flags) == 25


def test_presence_aware_loss_gradient():
    value = config()
    loss_function = build_loss(value)

    logits = torch.zeros(
        1,
        53,
        8,
        8,
        8,
        requires_grad=True,
    )
    target = torch.zeros(
        1,
        1,
        8,
        8,
        8,
        dtype=torch.long,
    )
    target[:, :, 3:5, 3:5, 3:5] = 7

    loss = loss_function(logits, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0
