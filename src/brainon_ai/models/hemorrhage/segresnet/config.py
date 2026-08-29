from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[5]


def default_config_path() -> Path:
    return REPOSITORY_ROOT / "configs" / "experiments" / "hemorrhage" / "H-S00.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path or default_config_path()).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"SegResNet 설정 파일이 없습니다: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    required = (
        "experiment",
        "data",
        "model",
        "loss",
        "optimizer",
        "scheduler",
        "training",
        "inference",
        "checkpoint",
        "pruning",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"SegResNet 설정 섹션이 없습니다: {missing}")
    _validate_config(config)
    config["_config_path"] = str(config_path)
    return config


def _validate_config(config: dict[str, Any]) -> None:
    manifest_value = str(config["experiment"].get("manifest_path", ""))
    if not manifest_value:
        raise ValueError("experiment.manifest_path가 비어 있습니다.")

    blocks_down = tuple(int(x) for x in config["model"]["blocks_down"])
    blocks_up = tuple(int(x) for x in config["model"]["blocks_up"])
    if len(blocks_up) != len(blocks_down) - 1:
        raise ValueError("model.blocks_up 길이는 blocks_down보다 1 작아야 합니다.")
    if int(config["model"]["init_filters"]) % 8 != 0:
        raise ValueError("SegResNet GroupNorm을 위해 init_filters는 8의 배수여야 합니다.")

    divisor = 2 ** (len(blocks_down) - 1)
    for key, value in (
        ("data.patch_size", config["data"]["patch_size"]),
        ("inference.patch_size", config["inference"]["patch_size"]),
    ):
        patch = tuple(int(x) for x in value)
        if len(patch) != 3 or min(patch) <= 0 or any(x % divisor for x in patch):
            raise ValueError(f"{key}는 양수 3개이고 각 축이 {divisor}의 배수여야 합니다: {patch}")

    positive_ints = {
        "data.batch_size": config["data"]["batch_size"],
        "data.patches_per_volume": config["data"]["patches_per_volume"],
        "training.epochs": config["training"]["epochs"],
        "training.validation_interval": config["training"]["validation_interval"],
    }
    for name, value in positive_ints.items():
        if int(value) < 1:
            raise ValueError(f"{name}는 1 이상이어야 합니다.")
    if str(config["pruning"].get("monitor", "composite")).lower() != "composite":
        raise ValueError("현재 pruning.monitor는 composite만 지원합니다.")
    weights = (
        float(config["checkpoint"]["dice_weight"]),
        float(config["checkpoint"]["recall_weight"]),
    )
    if min(weights) < 0 or sum(weights) <= 0:
        raise ValueError("checkpoint Dice/Recall 가중치는 음수가 아니고 합이 0보다 커야 합니다.")


def resolve_model_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def save_resolved_config(config: dict[str, Any], path: Path) -> None:
    payload = {key: value for key, value in config.items() if not key.startswith("_")}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)
