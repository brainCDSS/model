#!/usr/bin/env python3
"""Refine multimodal nnU-Net candidates and select by Dice plus lesion recall."""
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
import sys

ANEURYSM_VALIDATION = Path(__file__).resolve().parents[2] / "aneurysm_binary" / "validation"
sys.path.insert(0, str(ANEURYSM_VALIDATION))
from lesion_analysis import analyze_lesions
from report_i_n20_metrics import generate_report, summarize_lesion_sizes
from threshold_sweep import evaluate_thresholds
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
            "nnUNetTrainerIschemiaMultimodal__"
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
        f"{experiment['name']} Top-{keep_top_k} anchors={anchors}; "
        f"neighborhood={epochs}",
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
        "nnUNetTrainerIschemiaMultimodal",
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
        summarize_lesion_sizes(
            Path(lesion["metrics_csv"]),
            Path(lesion["size_summary_csv"]),
            config["evaluation"]["lesion_size_bins_ml"],
            config["evaluation"]["lesion_size_labels"],
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

    final_metrics_dir = fold / "final_metrics"
    generate_report(
        Path(best["summary"]),
        fold / "full_validation" / f"epoch_{int(best['epoch']):04d}",
        final_metrics_dir,
        float(config.get("evaluation", {}).get("surface_dice_tolerance_mm", 2.0)),
        config.get("evaluation", {}).get("lesion_size_bins_ml"),
        config.get("evaluation", {}).get("lesion_size_labels"),
    )
    # Keep the selected model's lesion-level table alongside final metrics.
    # This is deliberately separate from patient total-volume summaries.
    selected_lesions = fold / "lesion_analysis" / f"epoch_{int(best['epoch']):04d}" / "lesion_size_metrics.csv"
    if selected_lesions.is_file():
        shutil.copy2(selected_lesions, final_metrics_dir / "lesion_size_metrics.csv")
        summarize_lesion_sizes(
            selected_lesions,
            final_metrics_dir / "lesion_size_summary.csv",
            config["evaluation"]["lesion_size_bins_ml"],
            config["evaluation"]["lesion_size_labels"],
        )

    probability_config = config.get("probability_evaluation", {})
    if probability_config.get("enabled", False):
        probability_dir = fold / "best_probability_validation"
        expected_cases = len(
            json.loads(Path(best["summary"]).read_text(encoding="utf-8"))["metric_per_case"]
        )
        if len(list(probability_dir.glob("*.npz"))) != expected_cases:
            trainer.load_checkpoint(best["checkpoint"])
            transient = fold / "validation"
            if transient.exists():
                shutil.rmtree(transient)
            trainer.perform_actual_validation(True)
            if probability_dir.exists():
                shutil.rmtree(probability_dir)
            shutil.move(str(transient), str(probability_dir))
        threshold_report = evaluate_thresholds(
            Path(best["summary"]),
            probability_dir,
            final_metrics_dir,
            [float(value) for value in probability_config["thresholds"]],
            float(probability_config.get("small_lesion_max_ml", 0.1)),
            float(probability_config.get("recommended_max_fp_per_scan", 1.5)),
        )
        print(
            "threshold recommendation=",
            threshold_report["recommended_lesion_f2"]["threshold"],
            flush=True,
        )

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
