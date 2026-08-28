from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import yaml

from brainon_ai.common.config import save_config
from brainon_ai.common.runtime import (
    environment_info,
    git_commit,
    make_run_id,
    run_root,
    seed_everything,
    write_json,
)
from brainon_ai.metrics.segmentation import case_metrics
from brainon_ai.models.common.tiny3d import TinySegNet


def _load_cases(manifest: Path, split: str) -> list[dict[str, str]]:
    with manifest.open(encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["split"] == split]


def _load_case(row: dict[str, str]) -> tuple[torch.Tensor, torch.Tensor]:
    images = [nib.load(path).get_fdata(dtype=np.float32) for path in json.loads(row["image_paths"])]
    label = nib.load(row["label_path"]).get_fdata().astype(np.float32)
    return torch.from_numpy(np.stack(images))[None], torch.from_numpy(label)[None, None]


def _checkpoint_payload(
    model, optimizer, epoch: int, config: dict[str, Any], metrics: dict[str, float]
) -> dict[str, Any]:
    selection = config.get("selection", {})
    return {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": None,
        "amp_scaler_state": None,
        "validation_metrics": metrics,
        "primary_metric": selection.get("primary"),
        "secondary_metric": selection.get("secondary"),
        "threshold": config["evaluation"]["threshold"],
        "postprocessing": config["evaluation"].get("postprocessing", {}),
        "experiment_id": config["experiment_id"],
        "data_release": config["data_release"],
        "split_id": config.get("split_id", "synthetic-v1"),
        "seed": config["runtime"]["seed"],
        "git_commit": git_commit(),
        "environment": environment_info(),
    }


def train(config: dict[str, Any]) -> Path:
    seed = int(config["runtime"].get("seed", 42))
    seed_everything(seed)
    run_id = make_run_id(seed)
    directory = run_root() / config["task"] / config["experiment_id"] / run_id
    for child in ("logs", "metrics", "checkpoints/top_k", "predictions", "visualizations"):
        (directory / child).mkdir(parents=True, exist_ok=True)
    save_config(config, directory / "resolved_config.yaml")
    write_json(directory / "environment.json", environment_info())
    (directory / "git_commit.txt").write_text(git_commit(), encoding="utf-8")
    write_json(
        directory / "data_release.json",
        {"release": config["data_release"], "split_id": config.get("split_id")},
    )
    write_json(directory / "status.json", {"status": "running", "run_id": run_id})

    device = torch.device(
        "cuda" if torch.cuda.is_available() and config["runtime"].get("device") != "cpu" else "cpu"
    )
    model = TinySegNet(config["model"]["in_channels"], config["model"].get("features", 4)).to(
        device
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["training"].get("learning_rate", 1e-3))
    )
    start_epoch = 0
    resume = config["training"].get("resume")
    if resume:
        payload = torch.load(Path(resume), map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        start_epoch = payload["epoch"] + 1

    manifest = Path(config["data"]["manifest"])
    history = []
    best_loss = float("inf")
    patience = int(config["training"].get("early_stopping_patience", 5))
    stale = 0
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and config["runtime"].get("amp", True)
    )
    accumulation = int(config["training"].get("gradient_accumulation", 1))
    for epoch in range(start_epoch, int(config["training"].get("epochs", 1))):
        model.train()
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for step, row in enumerate(_load_cases(manifest, "train"), 1):
            inputs, targets = (item.to(device) for item in _load_case(row))
            with torch.amp.autocast(device.type, enabled=scaler.is_enabled()):
                loss = (
                    torch.nn.functional.binary_cross_entropy_with_logits(model(inputs), targets)
                    / accumulation
                )
            scaler.scale(loss).backward()
            if step % accumulation == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.item()) * accumulation)
        validation = evaluate_model(model, config, device)
        val_loss = validation.pop("loss")
        metrics = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": val_loss,
            **validation,
        }
        history.append(metrics)
        payload = _checkpoint_payload(model, optimizer, epoch, config, validation)
        torch.save(payload, directory / "checkpoints/last.ckpt")
        torch.save(payload, directory / f"checkpoints/top_k/epoch-{epoch:04d}.ckpt")
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            for name in ("best_primary.ckpt", "best_secondary.ckpt", "best_balanced.ckpt"):
                torch.save(payload, directory / "checkpoints" / name)
        else:
            stale += 1
            if stale >= patience:
                break
    write_json(directory / "metrics/summary.json", history[-1])
    with (directory / "metrics/history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    for name in (
        "per_case.csv",
        "per_lesion.csv",
        "per_source_dataset.csv",
        "per_lesion_size.csv",
        "threshold_sweep.csv",
    ):
        (directory / "metrics" / name).write_text("metric,value\nplaceholder,0\n", encoding="utf-8")
    write_json(directory / "status.json", {"status": "completed", "run_id": run_id})
    return directory


@torch.no_grad()
def evaluate_model(model, config: dict[str, Any], device: torch.device) -> dict[str, float]:
    model.eval()
    losses = []
    metrics = []
    for row in _load_cases(Path(config["data"]["manifest"]), "val"):
        inputs, targets = (item.to(device) for item in _load_case(row))
        logits = model(
            inputs
        )  # Small synthetic volumes fit whole-volume; production adapter uses sliding windows.
        losses.append(float(torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)))
        prediction = (
            (torch.sigmoid(logits) >= float(config["evaluation"]["threshold"])).cpu().numpy()[0, 0]
        )
        metrics.append(case_metrics(prediction, targets.cpu().numpy()[0, 0]))
    combined = {}
    for key in metrics[0]:
        values = np.asarray([item[key] for item in metrics], dtype=float)
        combined[key] = float(np.nanmean(values)) if not np.isnan(values).all() else float("nan")
    return {"loss": float(np.mean(losses)), **combined}


def evaluate_checkpoint(config: dict[str, Any], checkpoint: Path) -> dict[str, float]:
    device = torch.device("cpu")
    model = TinySegNet(config["model"]["in_channels"], config["model"].get("features", 4))
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=False)["model_state"]
    )
    return evaluate_model(model, config, device)


def find_run(run_id: str) -> Path:
    matches = list(run_root().glob(f"*/*/{run_id}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one run named {run_id}, found {len(matches)}")
    return matches[0]


def resume_run(run_id: str) -> Path:
    source = find_run(run_id)
    config = yaml.safe_load((source / "resolved_config.yaml").read_text(encoding="utf-8"))
    config["training"]["resume"] = str(source / "checkpoints/last.ckpt")
    return train(config)


def select_checkpoint(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
