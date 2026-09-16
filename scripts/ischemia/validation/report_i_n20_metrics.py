#!/usr/bin/env python3
"""Generate final hard-mask metrics for an ischemia nnU-Net selection."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

from brainon_ai.models.ischemia.segresnet.metrics import (
    calculate_lesion_metrics,
    calculate_surface_metrics,
    calculate_volume_metrics,
    calculate_voxel_metrics,
    get_surface,
)


def normalized_surface_dice(pred, gt, spacing, tolerance_mm):
    pred_surface, gt_surface = get_surface(pred), get_surface(gt)
    if not pred_surface.any() and not gt_surface.any():
        return 1.0
    if not pred_surface.any() or not gt_surface.any():
        return float("nan")
    to_gt = distance_transform_edt(~gt_surface, sampling=spacing)[pred_surface]
    to_pred = distance_transform_edt(~pred_surface, sampling=spacing)[gt_surface]
    return float(((to_gt <= tolerance_mm).sum() + (to_pred <= tolerance_mm).sum()) /
                 (len(to_gt) + len(to_pred)))
def summarize_lesion_sizes(
    metrics_csv: Path,
    output_csv: Path,
    bins_ml: list[float],
    labels: list[str],
) -> pd.DataFrame:
    """Summarize lesion Dice and detection rate in ischemia volume bins."""
    if len(labels) != len(bins_ml) + 1:
        raise ValueError("labels must contain len(bins_ml) + 1 entries")
    lesions = pd.read_csv(metrics_csv)
    lesions["size_group"] = pd.cut(
        lesions["gt_volume_ml"],
        bins=[-np.inf, *bins_ml, np.inf],
        labels=labels,
        right=False,
    )
    summary = (
        lesions.groupby("size_group", observed=False)
        .agg(
            gt_lesions=("lesion_id", "size"),
            detected_lesions=("detected", "sum"),
            detection_rate=("detected", "mean"),
            mean_lesion_dice=("lesion_dice", "mean"),
            median_lesion_dice=("lesion_dice", "median"),
            median_volume_ml=("gt_volume_ml", "median"),
        )
        .reset_index()
    )
    summary["detected/total"] = (
        summary["detected_lesions"].astype(int).astype(str)
        + "/" + summary["gt_lesions"].astype(int).astype(str)
    )
    summary.to_csv(output_csv, index=False)
    return summary


def generate_report(summary_path: Path, prediction_dir: Path, output_dir: Path, tolerance_mm=2.0):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = []
    for item in summary["metric_per_case"]:
        reference = Path(item["reference_file"])
        prediction = prediction_dir / reference.name
        pred_image, gt_image = nib.load(str(prediction)), nib.load(str(reference))
        pred = np.asanyarray(pred_image.dataobj) > 0
        gt = np.asanyarray(gt_image.dataobj) > 0
        spacing = tuple(float(x) for x in gt_image.header.get_zooms()[:3])
        row = {"case_id": reference.name.removesuffix(".nii.gz")}
        row.update(calculate_voxel_metrics(pred, gt))
        row.update(calculate_surface_metrics(pred, gt, spacing))
        row.update(calculate_volume_metrics(pred, gt, spacing))
        row.update(calculate_lesion_metrics(pred, gt))
        row["nsd_2mm"] = normalized_surface_dice(pred, gt, spacing, tolerance_mm)
        rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_case.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    numeric = {}
    for key in rows[0]:
        if key == "case_id":
            continue
        values = np.asarray([row[key] for row in rows], dtype=float)
        finite = values[np.isfinite(values)]
        numeric[key] = {
            "mean": float(finite.mean()) if finite.size else None,
            "median": float(np.median(finite)) if finite.size else None,
            "valid_cases": int(finite.size),
        }
    total_gt = sum(row["gt_lesion_count"] for row in rows)
    total_detected = sum(row["detected_gt_lesions"] for row in rows)
    total_pred = sum(row["pred_lesion_count"] for row in rows)
    total_matched = total_pred - sum(row["false_positive_lesions"] for row in rows)
    report = {
        "cases": len(rows),
        "surface_tolerance_mm": tolerance_mm,
        "macro": numeric,
        "micro_lesion_recall": total_detected / total_gt if total_gt else None,
        "micro_lesion_precision": total_matched / total_pred if total_pred else None,
        "fp_per_scan": sum(row["false_positive_lesions"] for row in rows) / len(rows),
        "complete_miss_cases": sum(row["gt_lesion_count"] > 0 and row["detected_gt_lesions"] == 0 for row in rows),
        "note": "Hard-mask metrics. PR-AUC/calibration require saved probability maps.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
