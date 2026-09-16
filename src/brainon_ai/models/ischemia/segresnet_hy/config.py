"""Configuration validation for isolated hy ischemia SegResNet experiments."""

from pathlib import Path
import re
import yaml


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    experiment_id = config.get("experiment_id")
    if not isinstance(experiment_id, str) or re.fullmatch(r"I-S\d{2,3}", experiment_id) is None:
        raise ValueError(f"experiment_id 형식 오류: {experiment_id!r}")
    if config.get("task") != "ischemia":
        raise ValueError("task는 ischemia여야 합니다")
    if config["data"].get("modalities") != ["DWI", "ADC"]:
        raise ValueError("입력 modality는 DWI, ADC여야 합니다")
    if config["data"].get("patch_size") != [80, 96, 80]:
        raise ValueError("patch_size는 비교 실험과 동일해야 합니다")
    if config["training"].get("epochs") != 300:
        raise ValueError("300 epoch 고정 실험입니다")
    if config["training"]["early_stopping"].get("enabled"):
        raise ValueError("early stopping은 비활성화해야 합니다")
