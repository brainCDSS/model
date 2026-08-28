from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.metadata
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_sha256(config: dict) -> str:
    payload = deepcopy({key: value for key, value in config.items() if not key.startswith("_")})
    # 장비/파일 위치가 바뀌어도 같은 학습 설정과 checkpoint를 사용할 수 있어야 합니다.
    for key in ("device", "output_dir", "manifest_path"):
        payload.get("experiment", {}).pop(key, None)
    payload.get("data", {}).pop("root_dir", None)
    payload.get("training", {}).pop("resume", None)
    payload.get("inference", {}).pop("save_predictions", None)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _versions() -> dict[str, str]:
    result = {"torch": torch.__version__}
    for package in ("monai", "nibabel", "pandas"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not-installed"
    return result


def _rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def build_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    global_step: int,
    best_metrics: dict[str, float],
    pruning_state: dict,
    config: dict,
    manifest_hash: str,
) -> dict:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metrics": {key: float(value) for key, value in best_metrics.items()},
        "pruning_state": dict(pruning_state),
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
        "config_sha256": config_sha256(config),
        "manifest_sha256": manifest_hash,
        "rng_state": _rng_state(),
        "versions": _versions(),
    }


def save_checkpoint(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint가 없습니다: {path}")
    return torch.load(path, map_location=device, weights_only=False)


def load_model_state(model: torch.nn.Module, checkpoint: dict) -> None:
    state = checkpoint.get("model_state", checkpoint.get("state_dict"))
    if state is None:
        raise KeyError("checkpoint에 model_state 또는 state_dict가 없습니다.")
    model.load_state_dict(state)
