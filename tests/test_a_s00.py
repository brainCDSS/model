from copy import deepcopy
import math

import torch

from brainon_ai.models.aneurysm.segresnet.checkpointing import (
    config_sha256,
)
from brainon_ai.models.aneurysm.segresnet.config import (
    load_config,
)
from brainon_ai.models.aneurysm.segresnet.data import (
    deterministic_indices,
)
from brainon_ai.models.aneurysm.segresnet.losses import (
    build_loss,
)
from brainon_ai.models.aneurysm.segresnet.metrics import (
    patient_metrics,
)
from brainon_ai.models.aneurysm.segresnet.network import (
    build_model,
)


def test_a_s00_contract():
    config = load_config()

    assert config["experiment_id"] == "A-S00"
    assert config["target"] == "location"
    assert config["data"]["patch_size"] == [
        112,
        160,
        64,
    ]
    assert config["data"]["batch_size"] == 1
    assert config["data"]["out_channels"] == 53
    assert config["data"]["use_type_masks"] is False
    assert config["data"]["split_seed"] == 20260828
    assert config["data"]["expected_train_cases"] == 215
    assert (
        config["data"]["expected_validation_cases"]
        == 46
    )
    assert config["data"]["fixed_test_cases"] == 46
    assert (
        config["data"]["expected_manifest_test_cases"]
        == 0
    )


def test_standard_segresnet_forward_and_loss():
    config = deepcopy(load_config())
    config["model"]["init_filters"] = 8
    config["model"]["blocks_down"] = [1, 1]
    config["model"]["blocks_up"] = [1]

    model = build_model(config).cpu()
    loss_function = build_loss(config)

    image = torch.randn(
        1,
        1,
        16,
        16,
        16,
    )
    label = torch.zeros(
        1,
        1,
        16,
        16,
        16,
        dtype=torch.long,
    )
    label[:, :, 4:8, 4:8, 4:8] = 5

    logits = model(image)

    assert tuple(logits.shape) == (
        1,
        53,
        16,
        16,
        16,
    )

    loss = loss_function(logits, label)

    assert torch.isfinite(loss)

    loss.backward()

    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
    )


def test_patient_metric_contract():
    target = torch.tensor(
        [0, 3, 3, 7, 7, 0],
        dtype=torch.int64,
    )
    prediction = torch.tensor(
        [0, 3, 0, 7, 5, 5],
        dtype=torch.int64,
    )

    row, _, _, _ = patient_metrics(
        prediction,
        target,
    )

    assert math.isclose(
        row["patient_dice"],
        0.75,
    )
    assert math.isclose(
        row["patient_macro_dice"],
        2.0 / 3.0,
    )

    empty = torch.zeros(
        8,
        dtype=torch.int64,
    )

    negative, _, _, _ = patient_metrics(
        empty,
        empty,
    )

    assert math.isnan(
        negative["patient_dice"]
    )
    assert math.isnan(
        negative["recall"]
    )


def test_unit_sampling_is_reproducible():
    arguments = {
        "case_count": 215,
        "batch_size": 2,
        "seed": 20260828,
        "epoch": 7,
        "iteration": 82,
        "phase": "train",
    }

    first = deterministic_indices(**arguments)
    second = deterministic_indices(**arguments)

    assert first == second

    validation = deterministic_indices(
        **{
            **arguments,
            "phase": "online_validation",
        }
    )

    assert validation != first


def test_config_hash_ignores_machine_paths():
    config = load_config()
    moved = deepcopy(config)

    moved["experiment"]["device"] = "cpu"
    moved["experiment"]["output_dir"] = (
        "/different/run"
    )
    moved["data"]["root_dir"] = (
        "/different/data"
    )
    moved["data"]["manifest_path"] = (
        "/different/manifest.csv"
    )
    moved["data"]["common_manifest_path"] = (
        "/different/common.csv"
    )
    moved["data"]["split_info_path"] = (
        "/different/split.json"
    )

    assert config_sha256(config) == config_sha256(
        moved
    )
