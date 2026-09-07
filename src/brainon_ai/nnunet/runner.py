from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from brainon_ai.nnunet.config import load_config, set_config_environment


TRAINER_NAME = "nnUNetTrainerICHBaseline"
CHECKPOINT_NAMES = {
    "best_composite": "checkpoint_best_composite.pth",
    "best_dice": "checkpoint_best_dice.pth",
    "best_recall": "checkpoint_best_recall.pth",
    "best": "checkpoint_best.pth",
    "final": "checkpoint_final.pth",
}


def configure_paths(data_root: str | Path, results_root: str | Path) -> dict[str, Path]:
    data_root = Path(data_root).resolve()
    results_root = Path(results_root).resolve()
    paths = {
        "nnUNet_raw": data_root / "nnUNet_raw",
        "nnUNet_preprocessed": data_root / "nnUNet_preprocessed",
        "nnUNet_results": results_root,
    }
    for name in ("nnUNet_raw", "nnUNet_preprocessed"):
        if not paths[name].is_dir():
            raise FileNotFoundError(f"{name} 폴더가 없습니다: {paths[name]}")
    results_root.mkdir(parents=True, exist_ok=True)
    for name, path in paths.items():
        os.environ[name] = str(path)
    trainer_path = Path(__file__).resolve().parent / "trainers"
    existing = os.environ.get("nnUNet_extTrainer", "")
    entries = [entry for entry in existing.split(os.pathsep) if entry]
    if str(trainer_path) not in entries:
        entries.insert(0, str(trainer_path))
    os.environ["nnUNet_extTrainer"] = os.pathsep.join(entries)
    return paths


def expected_output_folder(config: dict[str, Any], results_root: str | Path) -> Path:
    experiment = config["experiment"]
    dataset_name = f"Dataset{int(experiment['dataset_id']):03d}_ICHBinary"
    folder = (
        f"{TRAINER_NAME}__{config['plans']['output_identifier']}"
        f"__{experiment['configuration']}"
    )
    return Path(results_root).resolve() / dataset_name / folder / f"fold_{experiment['fold']}"


def resolve_device(requested: str):
    import torch

    requested = requested.lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in {"cuda", "cpu", "mps"}:
        raise ValueError(f"지원하지 않는 device입니다: {requested}")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda이지만 CUDA를 사용할 수 없습니다.")
    if requested == "cpu":
        torch.set_num_threads(os.cpu_count() or 1)
    elif requested == "cuda":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    return torch.device(requested)


def run_from_config(
    config_path: str | Path,
    data_root: str | Path,
    results_root: str | Path,
    *,
    run_validation: bool = True,
) -> Path:
    config_path = set_config_environment(config_path)
    config = load_config(config_path)
    paths = configure_paths(data_root, results_root)

    import torch
    from nnunetv2.run.run_training import get_trainer_from_args, maybe_load_checkpoint

    experiment = config["experiment"]
    device = resolve_device(str(experiment.get("device", "auto")))
    continue_training = bool(experiment.get("continue_training", False))
    trainer = get_trainer_from_args(
        str(experiment["dataset_id"]),
        str(experiment["configuration"]),
        int(experiment["fold"]),
        TRAINER_NAME,
        str(config["plans"]["output_identifier"]),
        continue_training,
        device=device,
    )
    maybe_load_checkpoint(trainer, continue_training, False)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    trainer.run_training()

    if run_validation:
        selected = str(experiment.get("validation_checkpoint", "final"))
        if selected not in CHECKPOINT_NAMES:
            choices = ", ".join(CHECKPOINT_NAMES)
            raise ValueError(f"validation_checkpoint는 다음 중 하나여야 합니다: {choices}")
        checkpoint = Path(trainer.output_folder) / CHECKPOINT_NAMES[selected]
        if not checkpoint.is_file():
            raise FileNotFoundError(f"검증 checkpoint가 없습니다: {checkpoint}")
        trainer.load_checkpoint(checkpoint)
        trainer.perform_actual_validation(False)
    return Path(trainer.output_folder)


def prepare_continuation_checkpoint(
    checkpoint: dict[str, Any], config: dict[str, Any], config_sha256: str
) -> dict[str, Any]:
    """Retarget a full training checkpoint to a new, traceable continuation run."""
    checkpoint["ich_config_sha256"] = config_sha256
    checkpoint["ich_config"] = {
        key: value for key, value in config.items() if not key.startswith("_")
    }
    checkpoint["_best_ema"] = None
    state = checkpoint.setdefault("ich_state", {})
    state.update(
        {
            "best_dice": float("-inf"),
            "best_recall": float("-inf"),
            "best_composite": float("-inf"),
            "best_full_macro_dice": float("-inf"),
            "best_full_macro_epoch": None,
            "pruning": {"best": float("-inf"), "bad_checks": 0},
        }
    )
    return checkpoint


def run_continuation_from_config(
    config_path: str | Path,
    parent_checkpoint: str | Path,
    data_root: str | Path,
    results_root: str | Path,
) -> Path:
    config_path = set_config_environment(config_path)
    config = load_config(config_path)
    configure_paths(data_root, results_root)

    import torch
    from nnunetv2.run.run_training import get_trainer_from_args

    experiment = config["experiment"]
    continuation = config.get("continuation")
    if not continuation:
        raise ValueError("continuation 설정 섹션이 없습니다.")
    parent_checkpoint = Path(parent_checkpoint).resolve()
    if not parent_checkpoint.is_file():
        raise FileNotFoundError(f"부모 checkpoint가 없습니다: {parent_checkpoint}")
    device = resolve_device(str(experiment.get("device", "auto")))
    trainer = get_trainer_from_args(
        str(experiment["dataset_id"]),
        str(experiment["configuration"]),
        int(experiment["fold"]),
        TRAINER_NAME,
        str(config["plans"]["output_identifier"]),
        False,
        device=device,
    )
    checkpoint = torch.load(parent_checkpoint, map_location="cpu", weights_only=False)
    expected_epoch = int(continuation["expected_parent_epoch"])
    actual_epoch = int(checkpoint.get("current_epoch", -1))
    if actual_epoch != expected_epoch:
        raise RuntimeError(
            f"부모 checkpoint epoch 불일치: expected={expected_epoch}, actual={actual_epoch}"
        )
    saved_plan = checkpoint.get("ich_config", {}).get("plans", {}).get("output_identifier")
    expected_plan = str(continuation["parent_output_identifier"])
    if saved_plan and saved_plan != expected_plan:
        raise RuntimeError(
            f"부모 plans 불일치: expected={expected_plan}, actual={saved_plan}"
        )
    checkpoint = prepare_continuation_checkpoint(
        checkpoint, config, trainer.ich_config_digest
    )
    output_folder = Path(trainer.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    epoch_folder = output_folder / "epoch_checkpoints"
    epoch_folder.mkdir(parents=True, exist_ok=True)
    initial_checkpoint = epoch_folder / f"checkpoint_epoch_{actual_epoch:04d}.pth"
    if not initial_checkpoint.is_file():
        temporary = initial_checkpoint.with_name(initial_checkpoint.name + ".tmp")
        torch.save(checkpoint, temporary)
        os.replace(temporary, initial_checkpoint)
    # nnU-Net 2.6.2 advertises dict support but only initializes its local
    # checkpoint variable for string paths. Load the retargeted file instead.
    trainer.load_checkpoint(str(initial_checkpoint))
    metadata = {
        "parent_experiment": continuation["parent_experiment"],
        "parent_checkpoint": str(parent_checkpoint),
        "parent_output_identifier": expected_plan,
        "parent_epoch": actual_epoch,
        "target_epoch": int(config["trainer"]["epochs"]),
        "optimizer_state_restored": True,
        "logger_history_restored": True,
        "early_stopping_reset": True,
        "early_stopping_enabled": bool(config["pruning"]["enabled"]),
    }
    (output_folder / "continuation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    trainer.run_training()
    return output_folder


def write_launch_summary(
    config_path: str | Path, data_root: str | Path, results_root: str | Path
) -> dict[str, Any]:
    config = load_config(config_path)
    paths = configure_paths(data_root, results_root)
    summary = {
        "config": str(Path(config_path).resolve()),
        "data_paths": {key: str(value) for key, value in paths.items()},
        "output_folder": str(expected_output_folder(config, results_root)),
        "experiment": config["experiment"],
        "plans": config["plans"],
        "trainer": config["trainer"],
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the H-N00 configurable nnU-Net baseline")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--results-root", default=Path("runs/nnunet"), type=Path)
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(write_launch_summary(args.config, args.data_root, args.results_root), indent=2))
        return
    output = run_from_config(
        args.config,
        args.data_root,
        args.results_root,
        run_validation=not args.no_validation,
    )
    print(output)


if __name__ == "__main__":
    main()
