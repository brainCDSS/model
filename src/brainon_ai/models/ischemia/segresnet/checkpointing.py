"""Checkpoint I/O compatible with the validated eight-key payload."""

import shutil
from pathlib import Path

import torch


CHECKPOINT_KEYS = (
    "epoch", "model", "optimizer", "scheduler", "scaler",
    "best_dice", "bad_epochs", "config",
)


def build_checkpoint(epoch, model, optimizer, scheduler, scaler, best_dice, bad_epochs, config):
    return {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_dice": best_dice,
        "bad_epochs": bad_epochs,
        "config": config,
    }


def save_checkpoint(state, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path("/content/segresnet_checkpoint_tmp.pth")
    torch.save(state, temp)
    shutil.copy2(temp, destination)
    temp.unlink(missing_ok=True)


def load_checkpoint(path: Path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    missing = [key for key in CHECKPOINT_KEYS if key not in checkpoint]
    if missing:
        raise KeyError(f"Checkpoint 필수 key 누락: {missing}")
    return checkpoint


def restore_training_state(checkpoint, model, optimizer, scheduler, scaler):
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint["best_dice"]),
        int(checkpoint["bad_epochs"]),
    )


def load_model_state(model, checkpoint):
    model.load_state_dict(checkpoint["model"], strict=True)
