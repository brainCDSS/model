#!/usr/bin/env python3
"""Refine A-B04 full-volume candidates and select by Dice plus lesion recall."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from brainon_ai.nnunet.config import set_config_environment
from lesion_analysis import analyze_lesions
from nnunetv2.run.run_training import get_trainer_from_args


def positive_macro(summary: dict) -> tuple[float, int]:
    values = []
    for item in summary.get("metric_per_case", []):
        metric = item["metrics"]["1"]
        value = float(metric["Dice"])
        if float(metric.get("n_ref", 0)) > 0 and np.isfinite(value):
            values.append(value)
    if not values:
        raise RuntimeError("positive validation case가 없습니다")
    return float(np.mean(values)), len(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    args = parser.parse_args()

    config_path = set_config_environment(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment, plans = config["experiment"], config["plans"]
    dataset = (
        f"Dataset{int(experiment['dataset_id']):03d}_{experiment['dataset_name']}"
    )
    fold = (
        args.results_root.resolve()
        / dataset
        / (
            "nnUNetTrainerAneurysmBinaryBaseline__"
            f"{plans['output_identifier']}__{experiment['configuration']}"
        )
        / f"fold_{experiment['fold']}"
    )
    history_path = fold / "full_validation_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    policy = config["full_validation"]
    keep_top_k = int(policy["keep_top_k"])
    radius = int(policy["neighborhood_radius"])
    anchors = [
        int(item["epoch"])
        for item in sorted(
            history,
            key=lambda item: float(item["macro_dice_positive"]),
            reverse=True,
        )[:keep_top_k]
    ]
    epochs = sorted(
        {
            epoch
            for anchor in anchors
            for epoch in range(
                max(1, anchor - radius),
                min(int(config["trainer"]["epochs"]), anchor + radius) + 1,
            )
        }
    )
    print(
        f"A-B04 Top-{keep_top_k} anchors={anchors}; neighborhood={epochs}",
        flush=True,
    )

    os.environ.update(
        nnUNet_raw=str(args.data_root.resolve() / "nnUNet_raw"),
        nnUNet_preprocessed=str(args.data_root.resolve() / "nnUNet_preprocessed"),
        nnUNet_results=str(args.results_root.resolve()),
    )
    trainer = get_trainer_from_args(
        str(experiment["dataset_id"]),
        experiment["configuration"],
        int(experiment["fold"]),
        "nnUNetTrainerAneurysmBinaryBaseline",
        plans["output_identifier"],
        False,
        device=torch.device("cuda"),
    )
    known = {int(item["epoch"]): item for item in history}
    for epoch in epochs:
        checkpoint = fold / "epoch_checkpoints" / f"checkpoint_epoch_{epoch:04d}.pth"
        if not checkpoint.is_file():
            print(f"epoch {epoch}: checkpoint 없음", flush=True)
            continue
        destination = fold / "full_validation" / f"epoch_{epoch:04d}"
        summary_path = destination / "summary.json"
        if not summary_path.is_file():
            trainer.load_checkpoint(checkpoint)
            transient = fold / "validation"
            if transient.exists():
                shutil.rmtree(transient)
            trainer.perform_actual_validation(False)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                shutil.rmtree(destination)
            shutil.move(str(transient), str(destination))
        macro_dice, positive_cases = positive_macro(
            json.loads(summary_path.read_text(encoding="utf-8"))
        )
        known[epoch] = {
            "epoch": epoch,
            "macro_dice_positive": macro_dice,
            "positive_cases": positive_cases,
            "summary": str(summary_path),
            "checkpoint": str(checkpoint),
        }
        print(epoch, f"macro Dice={macro_dice:.6f}", flush=True)

    history = [known[epoch] for epoch in sorted(known)]
    history_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    ranked = sorted(
        history,
        key=lambda item: float(item["macro_dice_positive"]),
        reverse=True,
    )
    best_macro = ranked[0]
    shutil.copy2(best_macro["checkpoint"], fold / "checkpoint_best_full_macro.pth")
    (fold / "best_full_macro.json").write_text(
        json.dumps(best_macro, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    selection = config.get("final_selection", {})
    lesion_candidate_count = int(selection.get("lesion_candidates", 3))
    tolerance = float(selection.get("macro_dice_tolerance", 0.01))
    candidates = []
    for item in ranked[:lesion_candidate_count]:
        epoch = int(item["epoch"])
        prediction_dir = fold / "full_validation" / f"epoch_{epoch:04d}"
        lesion_dir = fold / "lesion_analysis" / f"epoch_{epoch:04d}"
        lesion = analyze_lesions(
            Path(item["summary"]), prediction_dir, lesion_dir
        )
        candidate = {**item, **lesion}
        candidates.append(candidate)
        print(
            f"epoch {epoch}: macro Dice={item['macro_dice_positive']:.6f}, "
            f"lesion detection={lesion['detected_lesions']}/"
            f"{lesion['total_lesions']} "
            f"({lesion['lesion_detection_rate']:.4f})",
            flush=True,
        )

    eligible = [
        item
        for item in candidates
        if float(item["macro_dice_positive"])
        >= float(best_macro["macro_dice_positive"]) - tolerance
    ]
    best = max(
        eligible,
        key=lambda item: (
            float(item["lesion_detection_rate"]),
            float(item["macro_dice_positive"]),
        ),
    )
    final_selection = {
        "rule": (
            f"macro_dice_positive within {tolerance:.6f} of best; "
            "maximize lesion_detection_rate; tie-break by macro_dice_positive"
        ),
        "best_macro_dice": float(best_macro["macro_dice_positive"]),
        "candidates": candidates,
        "selected": best,
    }
    (fold / "final_selection.json").write_text(
        json.dumps(final_selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(best["checkpoint"], fold / "checkpoint_best_dual_criteria.pth")

    kept_epochs = {
        epoch
        for anchor in [int(item["epoch"]) for item in ranked[:keep_top_k]]
        for epoch in range(
            max(1, anchor - radius),
            min(int(config["trainer"]["epochs"]), anchor + radius) + 1,
        )
    }
    for checkpoint in (fold / "epoch_checkpoints").glob("checkpoint_epoch_*.pth"):
        if int(checkpoint.stem.rsplit("_", 1)[1]) not in kept_epochs:
            checkpoint.unlink()
    print(
        f"최종 선택 epoch={best['epoch']}, "
        f"macro Dice={best['macro_dice_positive']:.6f}, "
        f"lesion detection={best['lesion_detection_rate']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
