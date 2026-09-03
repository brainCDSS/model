from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path

import torch
from monai.data import DataLoader, Dataset, list_data_collate

from brainon_ai.models.ischemia.segresnet.checkpointing import (
    build_checkpoint,
    load_checkpoint,
    restore_training_state,
    save_checkpoint,
)
from brainon_ai.models.ischemia.segresnet.config import load_config
from brainon_ai.models.ischemia.segresnet.data import (
    build_records,
    train_transform,
    validation_transform,
)
from brainon_ai.models.ischemia.segresnet.inference import validate
from brainon_ai.models.ischemia.segresnet.losses import build_loss
from brainon_ai.models.ischemia.segresnet.network import build_model
from brainon_ai.models.ischemia.segresnet.pruning import EarlyStopping
from brainon_ai.models.ischemia.segresnet.reproducibility import seed_everything
from brainon_ai.models.ischemia.segresnet.sanity import run_sanity


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
    shutil.copy2(config_path, backup_root / "I-S00_effective_config.yaml")

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
    print("I-S00 SegResNet Baseline")
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
        transform=train_transform(patch_size, patches_per_volume),
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
    loss_fn = build_loss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    epochs = int(cfg["training"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp = bool(cfg["runtime"]["amp"])
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    if args.sanity_only:
        run_sanity(model, train_loader, loss_fn, device, amp)
        return

    latest_path = checkpoint_dir / "checkpoint_latest.pth"
    best_path = checkpoint_dir / "checkpoint_best.pth"
    start_epoch = 0

    early_stopping_cfg = cfg["training"]["early_stopping"]
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

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                output = model(image)
                loss = loss_fn(output, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())

        train_loss = running_loss / len(train_loader)
        val_dice = validate(
            model=model,
            loader=val_loader,
            device=device,
            roi_size=patch_size,
            sw_batch_size=sw_batch_size,
            overlap=overlap,
            threshold=threshold,
            amp=amp,
        )
        scheduler.step()
        improved = early_stopping.update(val_dice)

        state = build_checkpoint(
            epoch, model, optimizer, scheduler, scaler,
            early_stopping.best_dice, early_stopping.bad_epochs, cfg,
        )
        save_checkpoint(state, latest_path)
        if improved:
            save_checkpoint(state, best_path)

        epoch_seconds = time.time() - epoch_start
        lr = optimizer.param_groups[0]["lr"]
        with log_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [epoch + 1, train_loss, val_dice, early_stopping.best_dice, early_stopping.bad_epochs, lr, epoch_seconds]
            )

        print()
        print(f"Epoch {epoch + 1}/{epochs} 완료")
        print(f"Train Loss      : {train_loss:.4f}")
        print(f"Validation Dice : {val_dice:.4f}")
        print(f"Best Dice       : {early_stopping.best_dice:.4f}")
        print(f"Early Stopping  : {early_stopping.bad_epochs}/{early_stopping.patience}")
        print(f"Learning Rate   : {lr:.8f}")
        print(f"Epoch Time      : {epoch_seconds / 60:.1f} min")
        print("✓ checkpoint_latest.pth → Google Drive 백업")
        if improved:
            print("★ checkpoint_best.pth 갱신")
        if early_stopping.should_stop:
            print()
            print("Early Stopping triggered.")
            break

    print()
    print("=" * 60)
    print("I-S00 학습 종료")
    print(f"Best Validation Dice: {early_stopping.best_dice:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
