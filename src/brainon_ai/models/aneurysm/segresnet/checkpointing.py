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
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def config_sha256(config: dict) -> str:
    payload = deepcopy(
        {
            key: value
            for key, value in config.items()
            if not key.startswith("_")
        }
    )

    # Machine-specific execution paths are not model-contract fields.
    payload.get("experiment", {}).pop(
        "device",
        None,
    )
    payload.get("experiment", {}).pop(
        "output_dir",
        None,
    )

    for key in (
        "root_dir",
        "manifest_path",
        "common_manifest_path",
        "split_info_path",
    ):
        payload.get("data", {}).pop(
            key,
            None,
        )

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(
        encoded
    ).hexdigest()


def rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["torch_cuda"] = (
            torch.cuda.get_rng_state_all()
        )

    return state


def restore_rng_state(
    state: dict | None,
) -> None:
    if not state:
        return

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])

    if (
        torch.cuda.is_available()
        and "torch_cuda" in state
    ):
        torch.cuda.set_rng_state_all(
            state["torch_cuda"]
        )


def versions() -> dict[str, str]:
    result = {
        "torch": torch.__version__,
    }

    for package in (
        "monai",
        "numpy",
        "pandas",
        "nibabel",
    ):
        try:
            result[package] = (
                importlib.metadata.version(package)
            )
        except (
            importlib.metadata.PackageNotFoundError
        ):
            result[package] = "not-installed"

    return result


def build_checkpoint(
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    config: dict,
    manifest_hash: str,
    progress: dict,
) -> dict:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "config_sha256": config_sha256(config),
        "manifest_sha256": str(manifest_hash),
        "progress": deepcopy(progress),
        "rng_state": rng_state(),
        "versions": versions(),
    }


def atomic_torch_save(
    payload: dict,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name + ".tmp"
    )

    torch.save(
        payload,
        temporary,
    )

    os.replace(
        temporary,
        path,
    )


def atomic_json_save(
    payload: dict,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {path}"
        )

    return torch.load(
        path,
        map_location=device,
        weights_only=False,
    )


def validate_checkpoint(
    checkpoint: dict,
    config: dict,
    manifest_hash: str,
) -> None:
    current_config_hash = config_sha256(
        config
    )

    if (
        checkpoint.get("config_sha256")
        != current_config_hash
    ):
        raise RuntimeError(
            "Checkpoint config hash does not match "
            "the current A-S00 contract"
        )

    if (
        checkpoint.get("manifest_sha256")
        != str(manifest_hash)
    ):
        raise RuntimeError(
            "Checkpoint manifest hash does not match "
            "A-S00_manifest.csv"
        )


def load_training_state(
    model,
    optimizer,
    scheduler,
    scaler,
    checkpoint: dict,
) -> dict:
    model.load_state_dict(
        checkpoint["model_state"]
    )
    optimizer.load_state_dict(
        checkpoint["optimizer_state"]
    )
    scheduler.load_state_dict(
        checkpoint["scheduler_state"]
    )
    scaler.load_state_dict(
        checkpoint["scaler_state"]
    )

    restore_rng_state(
        checkpoint.get("rng_state")
    )

    return deepcopy(
        checkpoint["progress"]
    )
