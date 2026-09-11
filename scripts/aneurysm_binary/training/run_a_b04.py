#!/usr/bin/env python3
"""Launch/resume A-B04, then select by macro Dice and lesion recall."""
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
    experiment, plans = config["experiment"], config["plans"]
    data = (args.data_root or Path(config["paths"]["data_root"])).resolve()
    results = (args.results_root or Path(config["paths"]["results_root"])).resolve()
    python = Path(os.path.abspath(args.python_bin or repo / ".venv/bin/python"))
    train = python.parent / "nnUNetv2_train"
    trainers = repo / "src/brainon_ai/models/aneurysm_binary/nnunet"
    dataset = (
        f"Dataset{int(experiment['dataset_id']):03d}_{experiment['dataset_name']}"
    )
    preprocessed = data / "nnUNet_preprocessed"
    fold = (
        results
        / dataset
        / (
            "nnUNetTrainerAneurysmBinaryBaseline__"
            f"{plans['output_identifier']}__{experiment['configuration']}"
        )
        / f"fold_{experiment['fold']}"
    )
    required = {
        "Python": python,
        "nnUNetv2_train": train,
        "Trainer": trainers,
        "Dataset503": preprocessed / dataset,
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("필수 경로가 없습니다:\n" + "\n".join(missing))

    resume = (fold / "checkpoint_latest.pth").is_file() and not args.no_resume
    prepare = [
        str(python),
        str(repo / "scripts/aneurysm_binary/preprocessing/prepare_a_b00_plans.py"),
        "--config",
        str(config_path),
        "--data-root",
        str(data),
    ]
    command = [
        str(train),
        str(experiment["dataset_id"]),
        experiment["configuration"],
        str(experiment["fold"]),
        "-tr",
        "nnUNetTrainerAneurysmBinaryBaseline",
        "-p",
        plans["output_identifier"],
        "-device",
        experiment.get("device", "cuda"),
    ] + (["--c"] if resume else [])
    refine = [
        str(python),
        str(
            repo
            / "scripts/aneurysm_binary/validation/evaluate_a_b04_neighborhood.py"
        ),
        "--config",
        str(config_path),
        "--data-root",
        str(data),
        "--results-root",
        str(results),
    ]
    print(
        json.dumps(
            {
                "experiment": "A-B04",
                "mode": "resume" if resume else "new",
                "output_folder": str(fold),
                "prepare": prepare,
                "train": command,
                "post_training_selection": refine,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.dry_run:
        return 0

    results.mkdir(parents=True, exist_ok=True)
    subprocess.run(prepare, check=True)
    environment = os.environ.copy()
    environment.update(
        {
            "nnUNet_raw": str(data / "nnUNet_raw"),
            "nnUNet_preprocessed": str(preprocessed),
            "nnUNet_results": str(results),
            "nnUNet_extTrainer": str(trainers),
            "PYTHONPATH": os.pathsep.join(
                [str(repo / "src"), str(trainers), environment.get("PYTHONPATH", "")]
            ),
            "ICH_NNUNET_CONFIG": str(config_path),
            "nnUNet_compile": "false",
            "nnUNet_n_proc_DA": "0",
        }
    )
    code = subprocess.run(command, env=environment, check=False).returncode
    if code:
        return code
    return subprocess.run(refine, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
