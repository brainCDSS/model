#!/usr/bin/env python3
"""Launch/resume ischemia SegResNet experiment SegResNet on the nnU-Net fold-0-equivalent split."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--python-bin", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[3]
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = (args.data_root or repo.parent / "datasets/ischemic/SegResNet").resolve()
    results = (args.results_root or repo / "runs/I-segresnet").resolve()
    python = Path(args.python_bin or repo / ".venv/bin/python").absolute()
    output = results / config["experiment_id"]
    latest = output / "checkpoints/checkpoint_latest.pth"
    resume = latest.is_file() and not args.no_resume
    command = [
        str(python), "-m", "brainon_ai.models.ischemia.segresnet_hy.train",
        "--config", str(config_path), "--data-root", str(data),
        "--backup-root", str(output),
    ]
    print(json.dumps({
        "experiment": config["experiment_id"], "mode": "resume" if resume else "new",
        "data_root": str(data), "results": str(output), "command": command,
    }, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    if not python.is_file() or not (data / "train/images").is_dir():
        raise FileNotFoundError("Python 또는 SegResNet 데이터 경로를 확인하세요")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(repo / "src"), env.get("PYTHONPATH", "")])
    # The trainer resumes automatically when checkpoint_latest.pth is present.
    code = subprocess.run(command, cwd=repo, env=env, check=False).returncode
    if code:
        return code
    select = [
        str(python), str(repo / "scripts/ischemia/validation/select_i_s_neighborhood.py"),
        "--config", str(config_path), "--data-root", str(data),
        "--run-root", str(output), "--python-bin", str(python),
    ]
    return subprocess.run(select, cwd=repo, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
