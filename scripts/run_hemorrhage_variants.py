#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


EXPERIMENTS = ("H-N10", "H-N20", "H-S10", "H-S20")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def prepare_resenc_plans(data_root: Path, configs: dict[str, Path]) -> None:
    os.environ["nnUNet_raw"] = str(data_root / "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = str(data_root / "nnUNet_preprocessed")
    dataset_root = data_root / "nnUNet_preprocessed" / "Dataset001_ICHBinary"
    base = dataset_root / "nnUNetResEncUNetMPlans.json"
    if not base.is_file():
        from nnunetv2.experiment_planning.experiment_planners.residual_unets.residual_encoder_unet_planners import (
            nnUNetPlannerResEncM,
        )

        print("[prepare] nnUNet ResEnc M plans 생성")
        nnUNetPlannerResEncM(1).plan_experiment()

    for experiment in ("H-N10", "H-N20"):
        config = yaml.safe_load(configs[experiment].read_text(encoding="utf-8"))
        identifier = config["plans"]["output_identifier"]
        destination = dataset_root / f"{identifier}.json"
        if not destination.is_file():
            shutil.copy2(base, destination)
            payload = json.loads(destination.read_text(encoding="utf-8"))
            payload["plans_name"] = identifier
            destination.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )


def resumed_config(source: Path, destination: Path) -> Path:
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["experiment"]["continue_training"] = True
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return destination


def run_and_tee(command: list[str], environment: dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run] {' '.join(command)}")
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run four non-baseline hemorrhage experiments")
    parser.add_argument("--only", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[1]
    data_root = (args.data_root or (
        repository.parent / "datasets/hemorrhage/v1/model_ready/nnunet"
    )).resolve()
    segresnet_root = data_root.parent / "segresnet"
    manifest = data_root.parents[1] / "metadata/common/06_preprocessing_manifest.csv"
    required = [data_root / "nnUNet_raw", data_root / "nnUNet_preprocessed", segresnet_root, manifest]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        parser.error("필수 데이터 경로가 없습니다: " + ", ".join(missing))

    config_dir = repository / "configs/experiments/hemorrhage"
    configs = {name: config_dir / f"{name}.yaml" for name in EXPERIMENTS}
    run_root = repository / "runs/hemorrhage-variants"
    log_root = run_root / "logs"
    status_path = run_root / "batch_status.json"

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(repository / "src"), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    if not args.dry_run:
        prepare_resenc_plans(data_root, configs)

    commands: dict[str, list[str]] = {}
    for name in args.only:
        config_path = configs[name]
        if args.resume and name.startswith("H-N") and not args.dry_run:
            config_path = resumed_config(config_path, run_root / "resolved" / f"{name}.yaml")
        if name.startswith("H-N"):
            commands[name] = [
                sys.executable,
                "-m",
                "brainon_ai.nnunet.runner",
                "--config",
                str(config_path),
                "--data-root",
                str(data_root),
                "--results-root",
                str(repository / "runs/nnunet"),
            ]
        else:
            commands[name] = [
                sys.executable,
                "-m",
                "brainon_ai.models.hemorrhage.segresnet.train",
                "--config",
                str(config_path),
            ]
            if args.resume:
                commands[name].append("--resume")

    if args.dry_run:
        print(f"data_root={data_root}")
        print(f"segresnet_root={segresnet_root}")
        print(f"manifest={manifest}")
        for name, command in commands.items():
            print(f"{name}: {' '.join(command)}")
        return 0

    status = {"status": "running", "started_at": utc_now(), "experiments": {}}
    write_status(status_path, status)
    for name, command in commands.items():
        status["current"] = name
        status["experiments"][name] = {"status": "running", "started_at": utc_now()}
        write_status(status_path, status)
        code = run_and_tee(command, environment, log_root / f"{name}.log")
        status["experiments"][name].update(
            {"status": "completed" if code == 0 else "failed", "finished_at": utc_now(), "exit_code": code}
        )
        write_status(status_path, status)
        if code != 0:
            status.update({"status": "failed", "finished_at": utc_now()})
            write_status(status_path, status)
            return code

    status.update({"status": "completed", "finished_at": utc_now(), "current": None})
    write_status(status_path, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
