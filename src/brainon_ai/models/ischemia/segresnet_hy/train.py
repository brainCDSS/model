from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import torch
from monai.data import DataLoader, Dataset, list_data_collate

from brainon_ai.models.ischemia.segresnet_hy.checkpointing import (
    build_checkpoint,
    load_checkpoint,
    restore_training_state,
    save_checkpoint,
)
from brainon_ai.models.ischemia.segresnet_hy.config import load_config
from brainon_ai.models.ischemia.segresnet_hy.data import (
    build_records,
    train_transform,
    validation_transform,
)
from brainon_ai.models.ischemia.segresnet_hy.inference import validate
from brainon_ai.models.ischemia.segresnet_hy.losses import build_loss
from brainon_ai.models.ischemia.segresnet_hy.network import build_model
from brainon_ai.models.ischemia.segresnet_hy.pruning import EarlyStopping
from brainon_ai.models.ischemia.segresnet_hy.reproducibility import seed_everything
from brainon_ai.models.ischemia.segresnet_hy.sanity import run_sanity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--backup-root", required=True)
    parser.add_argument("--sanity-only", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    data_root = Path(args.data_root)
    backup_root = Path(args.backup_root)

    checkpoint_dir = backup_root / "checkpoints"
    log_dir = backup_root / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(config_path)
    experiment_id = cfg["experiment_id"]
    shutil.copy2(config_path, backup_root / f"{experiment_id}_effective_config.yaml")

    seed = int(cfg["runtime"]["seed"])
    seed_everything(seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 사용할 수 없습니다.")
    device = torch.device("cuda")

    patch_size = tuple(cfg["data"]["patch_size"])
    patches_per_volume = int(cfg["data"]["patches_per_volume"])
    batch_size = int(cfg["data"]["batch_size"])
    workers = int(cfg["data"]["num_workers"])

    train_records = build_records(data_root, "train")
    val_records = build_records(data_root, "val")
    test_records = build_records(data_root, "test")

    print("=" * 60)
    print(f"{experiment_id} SegResNet")
    print("=" * 60)
    print("GPU :", torch.cuda.get_device_name(0))
    print("Train:", len(train_records))
    print("Val  :", len(val_records))
    print("Test :", len(test_records))
    print("입력 : DWI + ADC (2 channels)")
    print("Patch:", patch_size)
    print("=" * 60)

    if (len(train_records), len(val_records), len(test_records)) != (175, 38, 37):
        raise RuntimeError("Train/Val/Test Case 수가 예상과 다릅니다.")

    train_ds = Dataset(
        train_records,
        transform=train_transform(patch_size, patches_per_volume, cfg.get("sampling")),
    )
    val_ds = Dataset(val_records, transform=validation_transform())

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        collate_fn=list_data_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    model = build_model(cfg).to(device)
    loss_fn = build_loss(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    epochs = int(cfg["training"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp = bool(cfg["runtime"]["amp"])
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    numerical = cfg.get("numerical_safety", {})
    gradient_clip = float(numerical.get("gradient_clip_max_norm", 0.0))
    fail_on_nonfinite = bool(numerical.get("fail_on_nonfinite", False))

    if args.sanity_only:
        run_sanity(model, train_loader, loss_fn, device, amp)
        return

    latest_path = checkpoint_dir / "checkpoint_latest.pth"
    best_path = checkpoint_dir / "checkpoint_best.pth"
    start_epoch = 0

    early_stopping_cfg = cfg["training"]["early_stopping"]
    early_stopping_enabled = bool(early_stopping_cfg.get("enabled", False))
    early_stopping = EarlyStopping(
        patience=int(early_stopping_cfg["patience"]),
        min_delta=float(early_stopping_cfg["min_delta"]),
    )

    if latest_path.is_file():
        print("기존 checkpoint 발견 → 학습 재개")
        checkpoint = load_checkpoint(latest_path, device)
        start_epoch, early_stopping.best_dice, early_stopping.bad_epochs = (
            restore_training_state(checkpoint, model, optimizer, scheduler, scaler)
        )
        print("Resume Epoch:", start_epoch)
        print("Best Dice:", early_stopping.best_dice)
        print("Early Stopping:", early_stopping.bad_epochs)

    log_path = log_dir / "training_log.csv"
    if not log_path.exists():
        with log_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                ["epoch", "train_loss", "val_dice", "best_dice", "bad_epochs", "learning_rate", "epoch_seconds"]
            )

    threshold = float(cfg["evaluation"]["threshold"])
    sw_batch_size = int(cfg["evaluation"]["sw_batch_size"])
    overlap = float(cfg["evaluation"]["overlap"])
    full_validation_cfg = cfg.get("full_validation", {})
    validation_start = int(full_validation_cfg.get("start_epoch", epochs + 1))
    validation_interval = int(full_validation_cfg.get("interval", 10))
    neighborhood_radius = int(full_validation_cfg.get("neighborhood_radius", 5))
    checkpoint_start = max(1, validation_start - neighborhood_radius)
    epoch_checkpoint_dir = checkpoint_dir / "epoch_checkpoints"
    history_path = backup_root / "full_validation_history.json"
    progress_path = backup_root / "progress.json"
    try:
        full_history = json.loads(history_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        full_history = []
    last_full_dice = full_history[-1]["macro_dice_positive"] if full_history else None
    numerics_path = log_dir / "numerics_log.csv"
    if not numerics_path.exists():
        with numerics_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(["epoch", "max_gradient_norm", "amp_scale"])

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        max_gradient_norm = 0.0

        for batch_index, batch in enumerate(train_loader, start=1):
            image = batch["image"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                output = model(image)
                loss = loss_fn(output, label)
            if fail_on_nonfinite and not torch.isfinite(loss):
                progress_path.write_text(json.dumps({
                    "status": "failed_nonfinite_loss", "current_epoch": epoch,
                    "batch_index": batch_index, "last_safe_checkpoint": str(latest_path),
                }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch + 1}, batch {batch_index}; "
                    f"last safe checkpoint: {latest_path}"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_limit = gradient_clip if gradient_clip > 0 else float("inf")
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_limit)
            if fail_on_nonfinite and not torch.isfinite(gradient_norm):
                optimizer.zero_grad(set_to_none=True)
                progress_path.write_text(json.dumps({
                    "status": "failed_nonfinite_gradient", "current_epoch": epoch,
                    "batch_index": batch_index, "last_safe_checkpoint": str(latest_path),
                }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                raise FloatingPointError(
                    f"Non-finite gradient at epoch {epoch + 1}, batch {batch_index}; "
                    f"last safe checkpoint: {latest_path}"
                )
            max_gradient_norm = max(max_gradient_norm, float(gradient_norm))
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())

        train_loss = running_loss / len(train_loader)
        completed_epoch = epoch + 1
        run_full_validation = (
            completed_epoch >= validation_start
            and completed_epoch % validation_interval == 0
        ) or completed_epoch == epochs
        val_dice = None
        improved = False
        if run_full_validation:
            val_dice = validate(
                model=model, loader=val_loader, device=device,
                roi_size=patch_size, sw_batch_size=sw_batch_size,
                overlap=overlap, threshold=threshold, amp=amp,
            )
            last_full_dice = float(val_dice)
            improved = early_stopping.update(val_dice)

        scheduler.step()
        state = build_checkpoint(
            epoch, model, optimizer, scheduler, scaler,
            early_stopping.best_dice, early_stopping.bad_epochs, cfg,
        )
        save_checkpoint(state, latest_path)
        if completed_epoch >= checkpoint_start:
            save_checkpoint(state, epoch_checkpoint_dir / f"checkpoint_epoch_{completed_epoch:04d}.pth")
        if improved:
            save_checkpoint(state, best_path)
        if run_full_validation:
            known = {int(item["epoch"]): item for item in full_history}
            known[completed_epoch] = {
                "epoch": completed_epoch,
                "macro_dice_positive": float(val_dice),
                "checkpoint": str(epoch_checkpoint_dir / f"checkpoint_epoch_{completed_epoch:04d}.pth"),
            }
            full_history = [known[key] for key in sorted(known)]
            history_path.write_text(json.dumps(full_history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        if completed_epoch >= checkpoint_start:
            anchors = [
                int(item["epoch"])
                for item in sorted(
                    full_history,
                    key=lambda item: float(item["macro_dice_positive"]),
                    reverse=True,
                )[: int(full_validation_cfg.get("keep_top_k", 3))]
            ]
            retained = {
                candidate
                for anchor in anchors
                for candidate in range(
                    max(checkpoint_start, anchor - neighborhood_radius),
                    min(epochs, anchor + neighborhood_radius) + 1,
                )
            }
            retained.update(
                range(max(checkpoint_start, completed_epoch - neighborhood_radius + 1), completed_epoch + 1)
            )
            for archived in epoch_checkpoint_dir.glob("checkpoint_epoch_*.pth"):
                archived_epoch = int(archived.stem.rsplit("_", 1)[1])
                if archived_epoch not in retained:
                    archived.unlink()

        epoch_seconds = time.time() - epoch_start
        lr = optimizer.param_groups[0]["lr"]
        with numerics_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow([completed_epoch, max_gradient_norm, scaler.get_scale() if amp else None])
        with log_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [completed_epoch, train_loss, val_dice, early_stopping.best_dice, early_stopping.bad_epochs, lr, epoch_seconds]
            )

        elapsed = time.time() - epoch_start
        progress = {
            "status": "completed" if completed_epoch == epochs else "running",
            "current_epoch": completed_epoch,
            "total_epochs": epochs,
            "progress_percent": round(100 * completed_epoch / epochs, 2),
            "eta_seconds": int(max(0, epochs - completed_epoch) * elapsed),
            "full_macro_dice": val_dice,
            "best_full_macro_dice": early_stopping.best_dice if early_stopping.best_dice >= 0 else None,
            "retained_epoch_checkpoints": len(list(epoch_checkpoint_dir.glob("checkpoint_epoch_*.pth"))),
            "max_gradient_norm": max_gradient_norm,
            "gradient_clip_max_norm": gradient_clip or None,
            "amp_enabled": amp,
        }
        progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        print()
        print(f"Epoch {epoch + 1}/{epochs} 완료")
        print(f"Train Loss      : {train_loss:.4f}")
        print(f"Full Val Dice   : {val_dice:.4f}" if val_dice is not None else "Full Val Dice   : scheduled full-volume validation")
        print(f"Best Full Dice  : {early_stopping.best_dice:.4f}" if early_stopping.best_dice >= 0 else "Best Full Dice  : -")
        print("Early Stopping  : disabled (fixed 300 epochs)")
        print(f"Learning Rate   : {lr:.8f}")
        print(f"Max Grad Norm   : {max_gradient_norm:.4f}")
        print(f"Epoch Time      : {epoch_seconds / 60:.1f} min")
        print(f"✓ checkpoint_latest.pth 저장: {latest_path}")
        if improved:
            print("★ checkpoint_best.pth 갱신")
        if early_stopping_enabled and early_stopping.should_stop:
            print()
            print("Early Stopping triggered.")
            break

    print()
    print("=" * 60)
    print(f"{experiment_id} 학습 종료")
    print(f"Best Validation Dice: {early_stopping.best_dice:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
