from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_overlay(
    image: np.ndarray, target: np.ndarray, prediction: np.ndarray, output: Path
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    slices = [image.shape[index] // 2 for index in range(3)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    views = [
        (image[slices[0]], target[slices[0]], prediction[slices[0]]),
        (image[:, slices[1]], target[:, slices[1]], prediction[:, slices[1]]),
        (image[:, :, slices[2]], target[:, :, slices[2]], prediction[:, :, slices[2]]),
    ]
    for axis, (base, truth, pred), title in zip(axes, views, ("sagittal", "coronal", "axial")):
        axis.imshow(np.rot90(base), cmap="gray")
        rgba = np.zeros((*truth.shape, 4))
        rgba[..., 0] = np.logical_and(pred, ~truth)
        rgba[..., 1] = np.logical_and(pred, truth)
        rgba[..., 2] = np.logical_and(~pred, truth)
        rgba[..., 3] = np.logical_or(pred, truth) * 0.55
        axis.imshow(np.rot90(rgba))
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=120)
    plt.close(fig)


def plot_run(run_dir: Path) -> list[Path]:
    history = run_dir / "metrics/history.csv"
    with history.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    epochs = [int(row["epoch"]) for row in rows]
    fig, axis = plt.subplots()
    axis.plot(epochs, [float(row["train_loss"]) for row in rows], label="train")
    axis.plot(epochs, [float(row["val_loss"]) for row in rows], label="validation")
    axis.legend()
    axis.set_xlabel("epoch")
    axis.set_ylabel("loss")
    output = run_dir / "visualizations/loss_curve.png"
    fig.savefig(output, dpi=120)
    plt.close(fig)
    (run_dir / "visualizations/loss_curve.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    return [output]
