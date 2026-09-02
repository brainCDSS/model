from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


CONFIG_ENV = "ICH_NNUNET_CONFIG"


def default_config_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "configs"
        / "experiments"
        / "hemorrhage"
        / "H-N00.yaml"
    )


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    candidate = Path(path or os.environ.get(CONFIG_ENV, default_config_path())).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"nnU-Net 설정 파일이 없습니다: {candidate}")
    with candidate.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    for section in ("experiment", "plans", "trainer", "loss", "checkpoint", "pruning", "network"):
        if section not in config:
            raise ValueError(f"nnU-Net 설정 섹션이 없습니다: {section}")
    if str(config["plans"]["base_identifier"]) == str(config["plans"]["output_identifier"]):
        raise ValueError("plans.output_identifier는 base_identifier와 달라야 합니다.")
    if str(config["pruning"].get("monitor", "composite")).lower() != "composite":
        raise ValueError("현재 pruning.monitor는 composite만 지원합니다.")
    weights = (
        float(config["checkpoint"]["dice_weight"]),
        float(config["checkpoint"]["recall_weight"]),
    )
    if min(weights) < 0 or sum(weights) <= 0:
        raise ValueError("checkpoint Dice/Recall 가중치는 음수가 아니고 합이 0보다 커야 합니다.")
    patch_size = config["plans"].get("patch_size")
    if not isinstance(patch_size, list) or len(patch_size) not in (2, 3):
        raise ValueError("plans.patch_size는 2D 또는 3D 정수 목록이어야 합니다.")
    if any(int(value) <= 0 for value in patch_size):
        raise ValueError("plans.patch_size의 모든 값은 양수여야 합니다.")
    if int(config["plans"].get("batch_size", 0)) <= 0:
        raise ValueError("plans.batch_size는 양수여야 합니다.")
    if int(config["trainer"].get("epochs", 0)) <= 0:
        raise ValueError("trainer.epochs는 양수여야 합니다.")
    sampling = config.get("sampling", {})
    if sampling.get("enabled", False):
        thresholds = sampling.get("volume_thresholds_ml")
        if not isinstance(thresholds, list) or len(thresholds) != 3:
            raise ValueError("sampling volume thresholds must contain three values")
        thresholds = [float(value) for value in thresholds]
        if thresholds != sorted(thresholds) or thresholds[0] <= 0:
            raise ValueError("sampling volume thresholds must be positive and sorted")
        strategy = str(sampling.get("strategy", "case_weighted")).lower()
        if strategy == "component_center":
            required = {"lt_1ml", "1_to_5ml", "5_to_20ml", "ge_20ml"}
            weights = sampling.get("component_weights", {})
        elif strategy == "case_weighted":
            required = {"negative", "lt_1ml", "1_to_5ml", "5_to_20ml", "ge_20ml"}
            weights = sampling.get("weights", {})
        else:
            raise ValueError(f"unsupported sampling.strategy: {strategy}")
        if not required.issubset(weights):
            raise ValueError("sampling weights are incomplete")
        if any(float(weights[name]) <= 0 for name in required):
            raise ValueError("sampling weights must be positive")
    config["_config_path"] = str(candidate)
    return config


def set_config_environment(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    os.environ[CONFIG_ENV] = str(resolved)
    return resolved


def config_digest(config: dict[str, Any]) -> str:
    payload = deepcopy({key: value for key, value in config.items() if not key.startswith("_")})
    for key in ("device", "continue_training", "validation_checkpoint"):
        payload.get("experiment", {}).pop(key, None)
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
