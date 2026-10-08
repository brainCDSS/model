from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from brainon_ai.common.config import load_config as load_yaml_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[5]


def default_config_path() -> Path:
    return (
        REPOSITORY_ROOT
        / "configs"
        / "experiments"
        / "aneurysm"
        / "A-S00.yaml"
    )



def resolve_repository_path(
    value: str | Path,
) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path.resolve()

    return (
        REPOSITORY_ROOT / path
    ).resolve()

def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path or default_config_path()).resolve()

    if not config_path.is_file():
        raise FileNotFoundError(
            f"A-S00 config does not exist: {config_path}"
        )

    config = load_yaml_config(config_path)
    _validate_config(config)
    config["_config_path"] = str(config_path)
    return config


def _validate_config(config: dict[str, Any]) -> None:
    required = (
        "task",
        "experiment_id",
        "target",
        "experiment",
        "data",
        "model",
        "loss",
        "optimizer",
        "scheduler",
        "training",
        "validation",
        "early_stopping",
        "resume",
        "inference",
        "completion",
    )

    missing = [key for key in required if key not in config]

    if missing:
        raise ValueError(
            f"A-S00 config sections are missing: {missing}"
        )

    if (
        config["task"] != "aneurysm"
        or config["target"] != "location"
    ):
        raise ValueError(
            "A-S00 must use task=aneurysm and target=location"
        )

    if config["experiment_id"] != "A-S00":
        raise ValueError(
            "This runner accepts only experiment_id=A-S00"
        )

    data = config["data"]
    model = config["model"]
    training = config["training"]
    online = config["validation"]["online_patch"]
    full_volume = config["validation"]["patient_full_volume"]
    resume = config["resume"]

    if bool(data["use_type_masks"]):
        raise ValueError("A-S00 must not use type masks")

    if int(data["split_seed"]) != 20260828:
        raise ValueError(
            "A-S00 split seed must remain 20260828"
        )

    expected_manifest_hash = (
        "f2b6e29cf3119a25828012b48ee15b8345fae3bb53b8e7ecbb619f0bd9768ffd"
    )
    expected_split_info_hash = (
        "b1ea64608af27973dd723e7b746ee41f26c5d54fbd47213097cc1f9c75d7f1e5"
    )
    expected_split_csv_hash = (
        "5f268b5231035934ec039ef00b62caff68751898cbd945b4b0b47dbc775466b4"
    )
    expected_split_json_hash = (
        "6e0b93727c9419bb107f6b74752102d81968ed1ff75d2a45ce9cfc8e75ba9279"
    )

    if data["common_manifest_sha256"] != expected_manifest_hash:
        raise ValueError(
            "A-S00 common dataset manifest SHA-256 does not match"
        )

    if data["split_info_sha256"] != expected_split_info_hash:
        raise ValueError(
            "A-S00 common split_info SHA-256 does not match"
        )

    if (
        data["frozen_split_source_csv_sha256"]
        != expected_split_csv_hash
    ):
        raise ValueError(
            "A-S00 frozen source CSV SHA-256 does not match"
        )

    if (
        data["frozen_split_source_json_sha256"]
        != expected_split_json_hash
    ):
        raise ValueError(
            "A-S00 frozen source JSON SHA-256 does not match"
        )

    expected_cases = (
        int(data["expected_train_cases"]),
        int(data["expected_validation_cases"]),
        int(data["fixed_test_cases"]),
    )

    if expected_cases != (215, 46, 46):
        raise ValueError(
            "A-S00 frozen split must be 215/46/46, "
            f"got {expected_cases}"
        )

    if int(data["expected_manifest_test_cases"]) != 0:
        raise ValueError(
            "A-S00 development manifest must contain zero test cases"
        )

    if (
        int(data["in_channels"]),
        int(data["out_channels"]),
    ) != (1, 53):
        raise ValueError(
            "A-S00 data channels must be 1 input and 53 outputs"
        )

    if (
        int(data["label_min"]),
        int(data["label_max"]),
    ) != (0, 52):
        raise ValueError(
            "A-S00 label range must remain 0..52"
        )

    if int(data["batch_size"]) < 1:
        raise ValueError(
            "A-S00 batch_size must be positive"
        )

    if (
        int(model["in_channels"]),
        int(model["out_channels"]),
    ) != (1, 53):
        raise ValueError(
            "A-S00 model channels must be 1 input and 53 outputs"
        )

    if str(model["name"]).lower() != "segresnet":
        raise ValueError(
            "A-S00 baseline must use the standard SegResNet model"
        )

    if int(model["init_filters"]) % 8:
        raise ValueError(
            "Standard SegResNet GroupNorm requires "
            "init_filters divisible by 8"
        )

    if (
        str(config["loss"]["name"]).lower()
        != "presence_aware_dice_ce"
    ):
        raise ValueError(
            "A-S00 requires presence_aware_dice_ce"
        )

    if bool(config["loss"]["include_background"]):
        raise ValueError(
            "A-S00 Dice must exclude background"
        )

    if float(data["foreground_oversample_fraction"]) != 0.5:
        raise ValueError(
            "A-S00 global foreground fraction must remain 0.5"
        )

    if (
        data["sampling_strategy"]
        != "deterministic_global_foreground"
    ):
        raise ValueError(
            "A-S00 sampling strategy must remain "
            "deterministic_global_foreground"
        )

    if float(online["foreground_fraction"]) != 0.5:
        raise ValueError(
            "A-S00 validation foreground fraction must remain 0.5"
        )

    if float(config["loss"]["background_ce_weight"]) != 0.001:
        raise ValueError(
            "A-S00 background CE weight must remain 0.001"
        )

    if float(config["loss"]["dice_weight"]) != 1.0:
        raise ValueError(
            "A-S00 Dice weight must remain 1.0"
        )

    if float(config["loss"]["ce_weight"]) != 0.5:
        raise ValueError(
            "A-S00 CE weight must remain 0.5"
        )

    blocks_down = tuple(
        int(value) for value in model["blocks_down"]
    )
    blocks_up = tuple(
        int(value) for value in model["blocks_up"]
    )

    if len(blocks_up) != len(blocks_down) - 1:
        raise ValueError(
            "model.blocks_up must have one fewer stage "
            "than blocks_down"
        )

    divisor = 2 ** (len(blocks_down) - 1)

    for name, values in (
        ("data.patch_size", data["patch_size"]),
        (
            "inference.roi_size",
            config["inference"]["roi_size"],
        ),
    ):
        patch = tuple(int(value) for value in values)

        if (
            len(patch) != 3
            or min(patch) <= 0
            or any(value % divisor for value in patch)
        ):
            raise ValueError(
                f"{name} must contain three positive "
                f"multiples of {divisor}: {patch}"
            )

    if tuple(int(v) for v in data["patch_size"]) != (
        112,
        160,
        64,
    ):
        raise ValueError(
            "A-S00 MONAI patch size must remain [112, 160, 64]"
        )

    if int(training["max_epochs"]) != 300:
        raise ValueError(
            "A-S00 max_epochs must remain 300"
        )

    if int(training["iterations_per_epoch"]) != 250:
        raise ValueError(
            "A-S00 train iterations must remain 250"
        )

    if (
        int(training["validation_iterations_per_epoch"])
        != 50
    ):
        raise ValueError(
            "A-S00 online validation iterations must remain 50"
        )

    if str(online["monitor"]) != "ema_fg_dice":
        raise ValueError(
            "A-S00 online monitor must be ema_fg_dice"
        )

    if not 0.0 < float(online["ema_alpha"]) < 1.0:
        raise ValueError(
            "validation.online_patch.ema_alpha must be in (0, 1)"
        )

    if bool(full_volume["every_epoch"]):
        raise ValueError(
            "Full-volume validation during training is forbidden"
        )

    if full_volume["schedule"] != "once_after_training":
        raise ValueError(
            "Full-volume validation must run once after training"
        )

    if int(full_volume["expected_cases"]) != 46:
        raise ValueError(
            "Final validation must contain 46 patients"
        )

    if full_volume["checkpoint"] != "checkpoint_best.pth":
        raise ValueError(
            "Final validation must use checkpoint_best.pth"
        )

    early_stopping = config["early_stopping"]

    if str(early_stopping["monitor"]) != "ema_fg_dice":
        raise ValueError(
            "Early stopping monitor must be ema_fg_dice"
        )

    if int(early_stopping["patience"]) != 80:
        raise ValueError(
            "Early stopping patience must remain 80"
        )

    if float(early_stopping["min_delta"]) != 0.0001:
        raise ValueError(
            "Early stopping min_delta must remain 0.0001"
        )

    if resume["mode"] != "strict_last_completed_unit":
        raise ValueError(
            "A-S00 requires strict_last_completed_unit resume"
        )

    for key in (
        "checkpoint_every_train_iterations",
        "checkpoint_every_validation_iterations",
        "checkpoint_every_final_patient",
    ):
        if int(resume[key]) != 1:
            raise ValueError(
                f"Strict resume requires resume.{key}=1"
            )

    if (
        int(resume["data_augmentation_workers"]) != 0
        or int(data["num_workers"]) != 0
    ):
        raise ValueError(
            "Strict resume requires data augmentation workers=0"
        )

    if (
        config["completion"]["marker"]
        != "final_validation/COMPLETE.json"
    ):
        raise ValueError(
            "A-S00 completion marker must be "
            "final_validation/COMPLETE.json"
        )


def save_resolved_config(
    config: dict[str, Any],
    path: Path,
) -> None:
    payload = {
        key: value
        for key, value in config.items()
        if not key.startswith("_")
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

    temporary.write_text(
        yaml.safe_dump(
            payload,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    temporary.replace(path)
