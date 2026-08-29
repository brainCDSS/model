from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

import pandas as pd
import torch
from tqdm.auto import tqdm

from brainon_ai.models.hemorrhage.segresnet.checkpointing import (
    build_checkpoint,
    config_sha256,
    load_checkpoint,
    load_model_state,
    restore_rng_state,
    save_checkpoint,
    sha256_file,
)
from brainon_ai.models.hemorrhage.segresnet.config import (
    load_config,
    resolve_model_path,
    save_resolved_config,
)
from brainon_ai.models.hemorrhage.segresnet.data import build_loaders, load_manifest
from brainon_ai.models.hemorrhage.segresnet.inference import evaluate_loader, save_evaluation
from brainon_ai.models.hemorrhage.segresnet.losses import build_loss
from brainon_ai.models.hemorrhage.segresnet.network import build_model
from brainon_ai.models.hemorrhage.segresnet.pruning import EarlyStopping
from brainon_ai.models.hemorrhage.segresnet.reproducibility import seed_everything


def _device(config: dict, override: str | None) -> torch.device:
    requested = str(override or config["experiment"].get("device", "cuda"))
    if requested.lower() == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 요청했지만 torch.cuda.is_available()이 False입니다.")
    return torch.device(requested)


def _optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    cfg = config["optimizer"]
    name = str(cfg["name"]).lower()
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg["learning_rate"]),
            weight_decay=float(cfg["weight_decay"]),
        )
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=float(cfg["learning_rate"]),
            weight_decay=float(cfg["weight_decay"]),
            momentum=float(cfg.get("momentum", 0.99)),
            nesterov=bool(cfg.get("nesterov", True)),
        )
    raise ValueError(f"지원하지 않는 optimizer: {name}")


def _scheduler(optimizer: torch.optim.Optimizer, config: dict):
    cfg = config["scheduler"]
    name = str(cfg["name"]).lower()
    if name != "cosine":
        raise ValueError(f"지원하지 않는 scheduler: {name}")
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["training"]["epochs"]),
        eta_min=float(cfg["min_learning_rate"]),
    )


def _grad_scaler(device: torch.device, enabled: bool):
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except TypeError:  # torch 2.3 계열 호환
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _train_epoch(
    model: torch.nn.Module,
    loader,
    loss_function,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    amp_enabled: bool,
    gradient_clip_norm: float,
    epoch: int,
) -> tuple[float, int]:
    model.train()
    running_loss = 0.0
    steps = 0
    progress = tqdm(loader, desc=f"train {epoch + 1}", leave=False)
    for batch in progress:
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(image)
            loss = loss_function(logits, label)
        if not torch.isfinite(loss):
            raise RuntimeError(f"epoch {epoch + 1}에서 유한하지 않은 loss가 발생했습니다: {loss.item()}")
        scaler.scale(loss).backward()
        if gradient_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        value = float(loss.detach())
        running_loss += value
        steps += 1
        progress.set_postfix(loss=f"{value:.4f}")
    if steps == 0:
        raise RuntimeError("train loader가 비어 있습니다.")
    return running_loss / steps, steps


def _append_history(path: Path, row: dict) -> None:
    frame = pd.DataFrame([row])
    frame.to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8-sig")


def _finite_or_none(value: float):
    return float(value) if math.isfinite(value) else None


def _write_progress(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _writer(log_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError:
        print("경고: tensorboard가 없어 TensorBoard 로그는 생략합니다.")
        return None
    return SummaryWriter(log_dir=str(log_dir))


def _resolve_resume(value: str | None, checkpoint_dir: Path) -> Path | None:
    if value in (None, "", "null", "none"):
        return None
    if str(value).lower() == "auto":
        return checkpoint_dir / "last.pt"
    return Path(value).resolve()


def run(args: argparse.Namespace) -> Path:
    config = load_config(args.config)
    if args.output_dir:
        config["experiment"]["output_dir"] = args.output_dir
    run_dir = resolve_model_path(config["experiment"]["output_dir"])
    checkpoint_dir = run_dir / "checkpoints"
    history_path = run_dir / "history.csv"
    manifest_path = resolve_model_path(config["experiment"]["manifest_path"])
    manifest_hash = sha256_file(manifest_path)
    resume_value = args.resume if args.resume is not None else config["training"].get("resume")
    resume_path = _resolve_resume(resume_value, checkpoint_dir)

    if history_path.exists() and resume_path is None:
        raise RuntimeError(
            f"기존 run을 덮어쓰지 않습니다: {run_dir}\n"
            "--resume auto를 사용하거나 config의 experiment.output_dir를 바꾸세요."
        )
    if args.weights and resume_path is not None:
        raise ValueError("--weights와 --resume은 동시에 사용할 수 없습니다.")

    device = _device(config, args.device)
    seed_everything(int(config["experiment"]["seed"]))
    data_root_value = config["data"].get("root_dir")
    data_root = resolve_model_path(data_root_value) if data_root_value else None
    manifest = load_manifest(manifest_path, data_root=data_root)
    loaders = build_loaders(config, manifest)
    model = build_model(config).to(device)
    loss_function = build_loss(config)
    optimizer = _optimizer(model, config)
    scheduler = _scheduler(optimizer, config)
    amp_enabled = bool(config["training"]["amp"] and device.type == "cuda")
    scaler = _grad_scaler(device, amp_enabled)
    pr_cfg = config["pruning"]
    pruner = EarlyStopping(
        enabled=bool(pr_cfg["enabled"]),
        min_epochs=int(pr_cfg["min_epochs"]),
        patience=int(pr_cfg["patience"]),
        min_delta=float(pr_cfg["min_delta"]),
    )

    start_epoch = 0
    global_step = 0
    best_metrics = {"dice": float("-inf"), "recall": float("-inf"), "composite": float("-inf")}
    if resume_path is not None:
        checkpoint = load_checkpoint(resume_path, device)
        if checkpoint.get("manifest_sha256") != manifest_hash:
            raise RuntimeError("resume checkpoint와 현재 manifest의 SHA-256이 다릅니다.")
        if checkpoint.get("config_sha256") != config_sha256(config):
            raise RuntimeError(
                "resume checkpoint와 현재 YAML 설정이 다릅니다. "
                "설정을 되돌리거나 새 experiment.output_dir에서 새 학습을 시작하세요."
            )
        load_model_state(model, checkpoint)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        best_metrics.update(checkpoint.get("best_metrics", {}))
        pruner.load_state_dict(checkpoint.get("pruning_state"))
        restore_rng_state(checkpoint.get("rng_state"))
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        print(f"학습 재개: {resume_path} (다음 epoch={start_epoch + 1})")
    elif args.weights:
        checkpoint = load_checkpoint(Path(args.weights).resolve(), device)
        load_model_state(model, checkpoint)
        print(f"가중치만 불러옴: {args.weights}")

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.json"
    save_resolved_config(config, run_dir / "config_resolved.yaml")
    manifest.to_csv(run_dir / "manifest_snapshot.csv", index=False, encoding="utf-8-sig")
    writer = _writer(run_dir / "tensorboard")

    epochs = int(config["training"]["epochs"])
    val_interval = int(config["training"]["validation_interval"])
    clip_norm = float(config["training"]["gradient_clip_norm"])
    checkpoint_delta = float(config["checkpoint"].get("min_delta", 0.0))
    print(f"device={device}, train={len(loaders['train'].dataset)}, validation={len(loaders['validation'].dataset)}")
    print(f"patch={config['data']['patch_size']}, batch={config['data']['batch_size']}, patches/volume={config['data']['patches_per_volume']}")

    session_started = monotonic()
    epoch_durations: list[float] = []
    completed_epochs = start_epoch
    should_stop = False
    last_train_loss = float("nan")
    last_metrics = {"dice": float("nan"), "recall": float("nan"), "composite": float("nan")}

    def update_progress(status: str, completed_epochs: int, message: str | None = None) -> None:
        mean_epoch_seconds = (
            sum(epoch_durations) / len(epoch_durations) if epoch_durations else None
        )
        gpu_memory_gb = None
        if device.type == "cuda":
            gpu_memory_gb = round(torch.cuda.max_memory_allocated(device) / 1024**3, 3)
        _write_progress(
            progress_path,
            {
                "status": status,
                "message": message,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "current_epoch": completed_epochs,
                "total_epochs": epochs,
                "progress_percent": round(100 * completed_epochs / epochs, 2),
                "session_elapsed_seconds": round(monotonic() - session_started, 2),
                "last_epoch_seconds": epoch_durations[-1] if epoch_durations else None,
                "mean_epoch_seconds": mean_epoch_seconds,
                "eta_seconds": (
                    mean_epoch_seconds * max(0, epochs - completed_epochs)
                    if mean_epoch_seconds
                    else None
                ),
                "train_loss": _finite_or_none(last_train_loss),
                "dice": _finite_or_none(last_metrics["dice"]),
                "recall": _finite_or_none(last_metrics["recall"]),
                "composite": _finite_or_none(last_metrics["composite"]),
                "best_dice": _finite_or_none(best_metrics["dice"]),
                "best_recall": _finite_or_none(best_metrics["recall"]),
                "best_composite": _finite_or_none(best_metrics["composite"]),
                "pruning_bad_checks": pruner.bad_checks,
                "pruning_patience": pruner.patience,
                "validation_interval": val_interval,
                "batch_size": int(config["data"]["batch_size"]),
                "patch_size": list(config["data"]["patch_size"]),
                "gpu_max_memory_gb": gpu_memory_gb,
                "output_folder": str(run_dir),
            },
        )

    update_progress("running", start_epoch)

    try:
        for epoch in range(start_epoch, epochs):
            epoch_started = monotonic()
            train_loss, steps = _train_epoch(
                model,
                loaders["train"],
                loss_function,
                optimizer,
                scaler,
                device,
                amp_enabled,
                clip_norm,
                epoch,
            )
            last_train_loss = train_loss
            global_step += steps
            scheduler.step()
            completed_epochs = epoch + 1
            row = {
                "epoch": completed_epochs,
                "train_loss": train_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "dice": float("nan"),
                "recall": float("nan"),
                "composite": float("nan"),
            }
            should_stop = False

            if completed_epochs % val_interval == 0 or completed_epochs == epochs:
                summary, per_case = evaluate_loader(
                    model,
                    loaders["validation"],
                    config,
                    device,
                    description=f"validation {completed_epochs}",
                )
                dice = float(summary["dice_positive_macro"])
                recall = float(summary["recall_positive_macro"])
                composite = float(summary["composite"])
                if not all(math.isfinite(value) for value in (dice, recall, composite)):
                    raise RuntimeError(f"validation 지표가 유한하지 않습니다: {summary}")
                row.update({"dice": dice, "recall": recall, "composite": composite})
                last_metrics.update({"dice": dice, "recall": recall, "composite": composite})
                improved = {
                    "dice": dice > best_metrics["dice"] + checkpoint_delta,
                    "recall": recall > best_metrics["recall"] + checkpoint_delta,
                    "composite": composite > best_metrics["composite"] + checkpoint_delta,
                }
                for name, value in (("dice", dice), ("recall", recall), ("composite", composite)):
                    if improved[name]:
                        best_metrics[name] = value
                should_stop = pruner.update(composite, completed_epochs)
                payload = build_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    global_step=global_step,
                    best_metrics=best_metrics,
                    pruning_state=pruner.state_dict(),
                    config=config,
                    manifest_hash=manifest_hash,
                )
                for name, changed in improved.items():
                    if changed:
                        save_checkpoint(payload, checkpoint_dir / f"best_{name}.pt")
                save_evaluation(summary, per_case, run_dir / "validation" / f"epoch_{completed_epochs:04d}")
                print(
                    f"epoch={completed_epochs} loss={train_loss:.4f} "
                    f"Dice={dice:.4f} Recall={recall:.4f} Composite={composite:.4f}"
                )
                if writer is not None:
                    writer.add_scalar("validation/dice_positive_macro", dice, completed_epochs)
                    writer.add_scalar("validation/recall_positive_macro", recall, completed_epochs)
                    writer.add_scalar("validation/composite", composite, completed_epochs)
            else:
                payload = build_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    global_step=global_step,
                    best_metrics=best_metrics,
                    pruning_state=pruner.state_dict(),
                    config=config,
                    manifest_hash=manifest_hash,
                )

            save_checkpoint(payload, checkpoint_dir / "last.pt")
            _append_history(history_path, row)
            epoch_durations.append(round(monotonic() - epoch_started, 3))
            update_progress(
                "early_stopping" if should_stop else "running",
                completed_epochs,
            )
            if writer is not None:
                writer.add_scalar("train/loss", train_loss, completed_epochs)
                writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], completed_epochs)
                writer.flush()
            if should_stop:
                print(
                    f"프루닝(조기 종료): composite가 validation {pruner.bad_checks}회 동안 "
                    f"{pruner.min_delta} 이상 개선되지 않았습니다."
                )
                break
    except BaseException as exc:
        update_progress("failed", max(start_epoch, locals().get("completed_epochs", start_epoch)), repr(exc))
        raise
    finally:
        if writer is not None:
            writer.close()

    update_progress("early_stopped" if should_stop else "completed", completed_epochs)

    print(f"완료: {run_dir}")
    print("최종 test는 학습과 분리되어 있습니다. ich-segresnet-evaluate --split test 를 실행하세요.")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="MONAI SegResNet ICH baseline 학습")
    parser.add_argument("--config", default=None, help="YAML 설정 경로")
    parser.add_argument("--device", default=None, help="auto, cuda, cuda:0 또는 cpu")
    parser.add_argument("--output-dir", default=None, help="model 폴더 기준 또는 절대 run 경로")
    parser.add_argument("--resume", nargs="?", const="auto", default=None, help="auto 또는 checkpoint 경로")
    parser.add_argument("--weights", default=None, help="가중치만 불러올 checkpoint 경로")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
