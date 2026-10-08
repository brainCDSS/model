#!/usr/bin/env python3
"""Select ischemia SegResNet experiment from Top-3 periodic anchors and their saved neighborhoods."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--python-bin", required=True, type=Path)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[3]
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    policy = config["full_validation"]
    top_k = int(policy["keep_top_k"])
    radius = int(policy["neighborhood_radius"])
    epochs = int(config["training"]["epochs"])
    history = pd.read_csv(args.run_root / "logs/training_log.csv")
    periodic = history[(history["epoch"] >= int(policy["start_epoch"])) & (history["epoch"] % int(policy["interval"]) == 0)]
    anchors = periodic.nlargest(top_k, "val_dice")["epoch"].astype(int).tolist()
    neighborhood = sorted({epoch for anchor in anchors for epoch in range(max(1, anchor-radius), min(epochs, anchor+radius)+1)})
    progress_path = args.run_root / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["status"] = "neighborhood_validation"
    progress["anchors"] = anchors
    progress["neighborhood_epochs"] = neighborhood
    progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(repo / "src"), env.get("PYTHONPATH", "")])
    evaluated = []
    evaluation_root = args.run_root / "selection_evaluation"
    for epoch in neighborhood:
        checkpoint = args.run_root / "checkpoints/epoch_checkpoints" / f"checkpoint_epoch_{epoch:04d}.pth"
        output = evaluation_root / f"epoch_{epoch:04d}"
        if not (output / "val_evaluation_metadata.json").is_file():
            if output.exists():
                shutil.rmtree(output)
            command = [str(args.python_bin), "-m", "brainon_ai.models.ischemia.segresnet_hy.evaluate", "--data-root", str(args.data_root), "--checkpoint", str(checkpoint), "--split", "val", "--output-dir", str(output)]
            subprocess.run(command, cwd=repo, env=env, check=True)
        cases = pd.read_csv(output / "val_positive_case_metrics.csv")
        total_gt = float(cases["gt_lesion_count"].sum())
        detected = float(cases["detected_gt_lesions"].sum())
        evaluated.append({"epoch": epoch, "macro_dice_positive": float(cases["dice"].mean()), "lesion_detection_rate": detected/total_gt if total_gt else 0.0, "checkpoint": str(checkpoint), "evaluation": str(output)})

    early_checkpoint = args.run_root / "checkpoints/checkpoint_best_preperiodic.pth"
    if early_checkpoint.is_file():
        import torch
        early_epoch = int(torch.load(early_checkpoint, map_location="cpu", weights_only=False)["epoch"]) + 1
        output = evaluation_root / f"epoch_{early_epoch:04d}_preperiodic"
        if not (output / "val_evaluation_metadata.json").is_file():
            if output.exists():
                shutil.rmtree(output)
            command = [str(args.python_bin), "-m", "brainon_ai.models.ischemia.segresnet_hy.evaluate", "--data-root", str(args.data_root), "--checkpoint", str(early_checkpoint), "--split", "val", "--output-dir", str(output)]
            subprocess.run(command, cwd=repo, env=env, check=True)
        cases = pd.read_csv(output / "val_positive_case_metrics.csv")
        total_gt = float(cases["gt_lesion_count"].sum())
        detected = float(cases["detected_gt_lesions"].sum())
        evaluated.append({"epoch": early_epoch, "macro_dice_positive": float(cases["dice"].mean()), "lesion_detection_rate": detected/total_gt if total_gt else 0.0, "checkpoint": str(early_checkpoint), "evaluation": str(output), "source": "preperiodic_best"})

    candidate_count = int(config["final_selection"]["lesion_candidates"])
    evaluated.sort(key=lambda item: item["macro_dice_positive"], reverse=True)
    candidates = evaluated[:candidate_count]
    best_macro = candidates[0]["macro_dice_positive"]
    tolerance = float(config["final_selection"]["macro_dice_tolerance"])
    eligible = [item for item in candidates if item["macro_dice_positive"] >= best_macro-tolerance]
    selected = max(eligible, key=lambda item: (item["lesion_detection_rate"], item["macro_dice_positive"]))
    final = {"rule": f"Top-{candidate_count} neighborhood models; macro Dice within {tolerance} of best; maximize lesion recall", "anchors": anchors, "neighborhood_epochs": neighborhood, "candidates": candidates, "selected": selected}
    (args.run_root / "final_selection.json").write_text(json.dumps(final, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    shutil.copy2(selected["checkpoint"], args.run_root / "checkpoints/checkpoint_best_dual_criteria.pth")

    final_metrics = args.run_root / "final_metrics"
    if final_metrics.exists():
        shutil.rmtree(final_metrics)
    command = [str(args.python_bin), "-m", "brainon_ai.models.ischemia.segresnet_hy.evaluate", "--data-root", str(args.data_root), "--checkpoint", selected["checkpoint"], "--split", "val", "--output-dir", str(final_metrics), "--save-probabilities"]
    subprocess.run(command, cwd=repo, env=env, check=True)
    sweep = [str(args.python_bin), str(repo / "scripts/ischemia/validation/evaluate_i_s_thresholds.py"), "--data-root", str(args.data_root), "--probability-dir", str(final_metrics / "val_probabilities"), "--output-dir", str(final_metrics), "--max-fp-per-scan", str(config["probability_evaluation"]["recommended_max_fp_per_scan"])]
    subprocess.run(sweep, cwd=repo, env=env, check=True)

    keep = set(neighborhood)
    for checkpoint in (args.run_root / "checkpoints/epoch_checkpoints").glob("checkpoint_epoch_*.pth"):
        if int(checkpoint.stem.rsplit("_", 1)[1]) not in keep:
            checkpoint.unlink()
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress.update({"status": "completed", "selected_epoch": selected["epoch"], "selected_macro_dice": selected["macro_dice_positive"], "selected_lesion_recall": selected["lesion_detection_rate"], "retained_epoch_checkpoints": len(keep)})
    progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(final, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
