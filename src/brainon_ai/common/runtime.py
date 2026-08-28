from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def repository_root() -> Path:
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return current


def cache_root() -> Path:
    return Path(os.getenv("BRAINON_CACHE_ROOT", repository_root() / ".cache/brainon")).resolve()


def run_root() -> Path:
    return Path(os.getenv("BRAINON_RUN_ROOT", repository_root() / "runs")).resolve()


def configure_nnunet() -> dict[str, str]:
    root = cache_root()
    values = {
        "nnUNet_raw": str(root / "nnUNet_raw"),
        "nnUNet_preprocessed": str(root / "nnUNet_preprocessed"),
        "nnUNet_results": str(root / "nnUNet_results"),
    }
    for name, value in values.items():
        os.environ[name] = value
        Path(value).mkdir(parents=True, exist_ok=True)
    return values


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "nogit"


def make_run_id(seed: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}_{git_commit()}_seed{seed}_{uuid.uuid4().hex[:8]}"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {},
    }
    try:
        import torch

        info.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except ImportError:
        info["torch"] = None
    return info


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
