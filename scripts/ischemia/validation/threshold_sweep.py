#!/usr/bin/env python3
"""Threshold sweep for nnU-Net foreground probability maps."""
from __future__ import annotations

import csv
import json
from itertools import permutations
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import label as connected_components


def _components(mask: np.ndarray):
    return connected_components(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))


def _foreground_probability(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        key = "probabilities" if "probabilities" in archive else "softmax" if "softmax" in archive else None
        if key is None:
            raise KeyError(f"{path}: probabilities/softmax key not found; keys={archive.files}")
        values = np.asarray(archive[key])
    if values.ndim != 4 or values.shape[0] < 2:
        raise ValueError(f"{path}: expected [classes, X, Y, Z], got {values.shape}")
    return values[1].astype(np.float32, copy=False)


def _align_probability_to_reference(
    probability: np.ndarray,
    reference_shape: tuple[int, ...],
    prediction_file: Path | None = None,
) -> np.ndarray:
    """Return a probability map in the reference NIfTI voxel order.

    nnU-Net .npz arrays use an internal array order, whereas the matching
    NIfTI uses nibabel's X/Y/Z order.  When dimensions have equal lengths,
    the saved 0.5 prediction mask resolves the otherwise ambiguous transpose.
    """
    if probability.shape == reference_shape:
        return probability

    candidates = [
        probability.transpose(axes)
        for axes in permutations(range(probability.ndim))
        if tuple(probability.transpose(axes).shape) == tuple(reference_shape)
    ]
    if not candidates:
        raise ValueError(
            f"Cannot align probability shape {probability.shape} to reference {reference_shape}"
        )
    if len(candidates) == 1:
        return candidates[0]
    if prediction_file is None or not prediction_file.is_file():
        raise ValueError(
            "Ambiguous probability axes; a saved prediction NIfTI is required "
            f"to align {probability.shape} to {reference_shape}"
        )

    saved_mask = np.asanyarray(nib.load(str(prediction_file)).dataobj) > 0
    if saved_mask.shape != tuple(reference_shape):
        raise ValueError(
            f"{prediction_file}: prediction shape {saved_mask.shape} != reference {reference_shape}"
        )
    agreement = [float(np.mean((candidate >= 0.5) == saved_mask)) for candidate in candidates]
    best_index = int(np.argmax(agreement))
    if agreement[best_index] < 0.99999:
        raise ValueError(
            f"Could not verify probability axes for {prediction_file}; best 0.5-mask agreement="
            f"{agreement[best_index]:.6f}"
        )
    return candidates[best_index]


def evaluate_thresholds(
    summary_path: Path,
    probability_dir: Path,
    output_dir: Path,
    thresholds: list[float],
    small_lesion_max_ml: float = 0.1,
    recommended_max_fp_per_scan: float = 1.5,
) -> dict:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cases = []
    for item in summary["metric_per_case"]:
        reference = Path(item["reference_file"])
        gt_image = nib.load(str(reference))
        gt = np.asanyarray(gt_image.dataobj) > 0
        probability_path = probability_dir / f"{reference.name.removesuffix('.nii.gz')}.npz"
        prediction_file = Path(item["prediction_file"]) if item.get("prediction_file") else None
        # nnU-Net summary paths point at the transient validation directory,
        # which is moved to ``best_probability_validation`` after export.
        if prediction_file is None or not prediction_file.is_file():
            exported_prediction = probability_dir / reference.name
            prediction_file = exported_prediction if exported_prediction.is_file() else prediction_file
        probability = _align_probability_to_reference(
            _foreground_probability(probability_path), gt.shape, prediction_file
        )
        voxel_volume_ml = float(np.prod(gt_image.header.get_zooms()[:3])) / 1000.0
        gt_components, gt_count = _components(gt)
        small_ids = [
            component_id
            for component_id in range(1, gt_count + 1)
            if int((gt_components == component_id).sum()) * voxel_volume_ml < small_lesion_max_ml
        ]
        cases.append((probability, gt, gt_components, gt_count, small_ids))

    # Probability quality is aggregated voxel-wise without materializing the
    # entire validation cohort in memory. PR-AUC is a 100-bin approximation.
    probability_bin_count = np.zeros(100, dtype=np.int64)
    probability_bin_sum = np.zeros(100, dtype=np.float64)
    probability_target_sum = np.zeros(100, dtype=np.float64)
    brier_sum = 0.0
    voxel_count = positive_count = 0
    for probability, gt, *_ in cases:
        values = probability.ravel().astype(np.float64, copy=False)
        targets = gt.ravel().astype(np.float64, copy=False)
        indices = np.minimum((values * 100).astype(np.int64), 99)
        probability_bin_count += np.bincount(indices, minlength=100)
        probability_bin_sum += np.bincount(indices, weights=values, minlength=100)
        probability_target_sum += np.bincount(indices, weights=targets, minlength=100)
        brier_sum += float(np.square(values - targets).sum())
        voxel_count += int(values.size)
        positive_count += int(targets.sum())
    occupied = probability_bin_count > 0
    confidence = np.divide(probability_bin_sum, probability_bin_count, where=occupied, out=np.zeros(100))
    observed = np.divide(probability_target_sum, probability_bin_count, where=occupied, out=np.zeros(100))
    ece = float(np.sum(probability_bin_count[occupied] * np.abs(confidence[occupied] - observed[occupied])) / voxel_count)
    tp_desc = np.cumsum(probability_target_sum[::-1])
    pred_desc = np.cumsum(probability_bin_count[::-1])
    recall_desc = tp_desc / positive_count if positive_count else np.zeros_like(tp_desc)
    precision_desc = np.divide(tp_desc, pred_desc, where=pred_desc > 0, out=np.zeros_like(tp_desc))
    pr_auc_approx = float(np.sum(precision_desc * np.diff(np.r_[0.0, recall_desc]))) if positive_count else None
    probability_quality = {
        "voxel_count": voxel_count, "positive_voxel_count": positive_count,
        "brier_score": brier_sum / voxel_count if voxel_count else None,
        "ece_100_bins": ece, "pr_auc_100_bin_approx": pr_auc_approx,
        "note": "Voxel-weighted; PR-AUC is a histogram approximation to avoid cohort-scale array concatenation.",
    }

    rows = []
    for threshold in thresholds:
        dice_values, recalls, precisions = [], [], []
        total_gt = total_detected = total_pred = total_fp = 0
        total_small = detected_small = 0
        for probability, gt, gt_components, gt_count, small_ids in cases:
            pred = probability >= float(threshold)
            tp = int(np.logical_and(pred, gt).sum())
            fp = int(np.logical_and(pred, ~gt).sum())
            fn = int(np.logical_and(~pred, gt).sum())
            denominator = 2 * tp + fp + fn
            dice_values.append(2 * tp / denominator if denominator else 1.0)
            recalls.append(tp / (tp + fn) if tp + fn else 1.0)
            precisions.append(tp / (tp + fp) if tp + fp else 1.0)
            pred_components, pred_count = _components(pred)
            detected = int(sum(
                bool(np.any(pred[gt_components == component_id]))
                for component_id in range(1, gt_count + 1)
            ))
            false_positives = int(sum(
                not bool(np.any(gt[pred_components == component_id]))
                for component_id in range(1, pred_count + 1)
            ))
            total_gt += gt_count
            total_detected += detected
            total_pred += pred_count
            total_fp += false_positives
            total_small += len(small_ids)
            detected_small += int(sum(
                bool(np.any(pred[gt_components == component_id]))
                for component_id in small_ids
            ))
        lesion_recall = total_detected / total_gt if total_gt else 1.0
        matched_predictions = total_pred - total_fp
        lesion_precision = matched_predictions / total_pred if total_pred else 1.0
        beta2 = 4.0
        lesion_f1 = (
            2 * lesion_precision * lesion_recall / (lesion_precision + lesion_recall)
            if lesion_precision + lesion_recall else 0.0
        )
        lesion_f2 = (
            (1 + beta2) * lesion_precision * lesion_recall
            / (beta2 * lesion_precision + lesion_recall)
            if lesion_precision + lesion_recall else 0.0
        )
        rows.append({
            "threshold": float(threshold),
            "macro_dice": float(np.mean(dice_values)),
            "voxel_recall": float(np.mean(recalls)),
            "voxel_precision": float(np.mean(precisions)),
            "lesion_recall": lesion_recall,
            "lesion_precision": lesion_precision,
            "lesion_f1": lesion_f1,
            "lesion_f2": lesion_f2,
            "fp_per_scan": total_fp / len(cases),
            "small_lesion_recall": detected_small / total_small if total_small else None,
            "detected_lesions": total_detected,
            "total_gt_lesions": total_gt,
            "false_positive_lesions": total_fp,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "threshold_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best_dice = max(rows, key=lambda row: row["macro_dice"])
    eligible = [row for row in rows if row["fp_per_scan"] <= recommended_max_fp_per_scan]
    recommended = max(eligible or rows, key=lambda row: (row["lesion_f2"], row["macro_dice"]))
    default = min(rows, key=lambda row: abs(row["threshold"] - 0.5))
    (output_dir / "probability_quality.json").write_text(
        json.dumps(probability_quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "probability_dir": str(probability_dir),
        "probability_quality": probability_quality,
        "small_lesion_max_ml": small_lesion_max_ml,
        "recommended_max_fp_per_scan": recommended_max_fp_per_scan,
        "default_0_5": default,
        "best_macro_dice": best_dice,
        "recommended_lesion_f2": recommended,
        "rows": rows,
    }
    (output_dir / "threshold_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
