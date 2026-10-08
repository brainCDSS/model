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


def _write_summary_tables(rows: list[dict], output_dir: Path, bins_ml: list[float], labels: list[str]) -> dict:
    """Write SegResNet-comparable case, size-group and negative-case summaries."""
    frame = pd.DataFrame(rows)
    numeric_rows = []
    for column in frame.select_dtypes(include=[np.number]).columns:
        values = frame[column].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        if not len(values):
            continue
        numeric_rows.append({
            "metric": column, "n_valid": int(len(values)), "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "median": float(np.median(values)), "q1": float(np.quantile(values, .25)),
            "q3": float(np.quantile(values, .75)), "min": float(values.min()), "max": float(values.max()),
        })
    pd.DataFrame(numeric_rows).to_csv(output_dir / "summary_metrics.csv", index=False)

    positive = frame[frame["gt_lesion_count"] > 0].copy()
    if not positive.empty:
        positive["size_group"] = pd.cut(
            positive["gt_volume_ml"], bins=[-np.inf, *bins_ml, np.inf],
            labels=labels, right=False,
        )
        size_rows = []
        for group in labels:
            group_cases = positive[positive["size_group"] == group]
            total_gt = int(group_cases["gt_lesion_count"].sum())
            detected = int(group_cases["detected_gt_lesions"].sum())
            size_rows.append({
                "size_group": group, "n_cases": int(len(group_cases)),
                "median_gt_volume_ml": float(group_cases["gt_volume_ml"].median()) if len(group_cases) else None,
                "macro_dice": float(group_cases["dice"].mean()) if len(group_cases) else None,
                "macro_iou": float(group_cases["iou"].mean()) if len(group_cases) else None,
                "macro_precision": float(group_cases["precision"].mean()) if len(group_cases) else None,
                "macro_recall_sensitivity": float(group_cases["recall_sensitivity"].mean()) if len(group_cases) else None,
                "macro_hd95_mm": float(group_cases["hd95_mm"].mean()) if len(group_cases) else None,
                "macro_assd_mm": float(group_cases["assd_mm"].mean()) if len(group_cases) else None,
                "gt_lesions": total_gt, "detected_lesions": detected,
                "lesion_detection_rate": detected / total_gt if total_gt else None,
                "fp_per_scan": float(group_cases["false_positive_lesions"].mean()) if len(group_cases) else None,
            })
        pd.DataFrame(size_rows).to_csv(output_dir / "case_volume_summary.csv", index=False)

    negative = frame[frame["gt_lesion_count"] == 0].copy()
    if negative.empty:
        negative = pd.DataFrame(columns=["case_id", "pred_lesion_count", "false_positive_lesions"])
    negative["negative_case_fp"] = negative.get("pred_lesion_count", pd.Series(dtype=float)) > 0
    negative.to_csv(output_dir / "negative_case_fp.csv", index=False)
    return {"positive_cases": int(len(positive)), "negative_cases": int(len(negative))}


def generate_report(
    summary_path: Path, prediction_dir: Path, output_dir: Path, tolerance_mm=2.0,
    lesion_size_bins_ml: list[float] | None = None, lesion_size_labels: list[str] | None = None,
):
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
    bins = lesion_size_bins_ml or [1.0, 5.0, 20.0]
    labels = lesion_size_labels or ["<1 mL", "1–5 mL", "5–20 mL", "≥20 mL"]
    cohort = _write_summary_tables(rows, output_dir, bins, labels)

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
        **cohort,
        "note": "Hard-mask metrics. PR-AUC/calibration require saved probability maps.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
