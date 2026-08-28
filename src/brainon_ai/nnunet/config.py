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
    return Path(__file__).resolve().parents[2] / "configs" / "baseline.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    candidate = Path(path or os.environ.get(CONFIG_ENV, default_config_path())).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"nnU-Net 설정 파일이 없습니다: {candidate}")
    with candidate.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    for section in ("experiment", "plans", "trainer", "loss", "checkpoint", "pruning", "network"):
        if section not in config:
            raise ValueError(f"설정 섹션이 없습니다: {section}")
    if str(config["plans"]["base_identifier"]) == str(config["plans"]["output_identifier"]):
        raise ValueError("plans.output_identifier는 base_identifier와 달라야 합니다.")
    if str(config["pruning"].get("monitor", "composite")).lower() != "composite":
        raise ValueError("현재 pruning.monitor는 composite만 지원합니다.")
    weights = (float(config["checkpoint"]["dice_weight"]), float(config["checkpoint"]["recall_weight"]))
    if min(weights) < 0 or sum(weights) <= 0:
        raise ValueError("checkpoint Dice/Recall 가중치는 음수가 아니며 합이 0보다 커야 합니다.")
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
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
