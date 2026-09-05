"""Load and validate effective ischemia SegResNet configurations."""

from pathlib import Path
import re

import yaml


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    try:
        experiment_id = config["experiment_id"]
    except (KeyError, TypeError) as exc:
        raise ValueError("필수 설정 누락: experiment_id") from exc

    if not isinstance(experiment_id, str) or re.fullmatch(r"I-S\d{2}", experiment_id) is None:
        raise ValueError(
            f"ischemia SegResNet experiment_id 형식 오류: {experiment_id!r}"
        )

    try:
        model_name = config["model"]["name"]
    except (KeyError, TypeError) as exc:
        raise ValueError("필수 설정 누락: model.name") from exc
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("필수 설정 오류: model.name")

    expected = {
        ("task",): "ischemia",
        ("data", "modalities"): ["DWI", "ADC"],
        ("data", "in_channels"): 2,
        ("data", "patch_size"): [80, 96, 80],
        ("data", "batch_size"): 1,
        ("data", "patches_per_volume"): 2,
        ("model", "init_filters"): 16,
        ("model", "out_channels"): 1,
        ("training", "epochs"): 300,
        ("training", "learning_rate"): 0.001,
        ("training", "weight_decay"): 0.00001,
        ("training", "early_stopping", "enabled"): True,
        ("training", "early_stopping", "patience"): 30,
        ("training", "early_stopping", "min_delta"): 0.001,
        ("runtime", "seed"): 42,
        ("runtime", "amp"): True,
        ("evaluation", "threshold"): 0.5,
        ("evaluation", "sw_batch_size"): 2,
        ("evaluation", "overlap"): 0.25,
    }
    for path, wanted in expected.items():
        value = config
        try:
            for key in path:
                value = value[key]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"필수 설정 누락: {'.'.join(path)}") from exc
        if value != wanted:
            raise ValueError(
                f"설정 불일치: {'.'.join(path)}={value!r}, expected {wanted!r}"
            )

    workers = config["data"].get("num_workers")
    if workers is None or int(workers) < 0:
        raise ValueError("필수 설정 오류: data.num_workers")
