from __future__ import annotations

import numpy as np
from scipy import ndimage


def dice_fg(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction, target = prediction.astype(bool), target.astype(bool)
    denominator = prediction.sum() + target.sum()
    return float(2 * np.logical_and(prediction, target).sum() / denominator) if denominator else 1.0


def components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    return ndimage.label(mask.astype(bool), structure=np.ones((3, 3, 3), dtype=np.uint8))


def lesion_metrics(
    prediction: np.ndarray, target: np.ndarray, overlap_threshold: float = 0.1
) -> dict[str, float]:
    pred_labels, pred_count = components(prediction)
    target_labels, target_count = components(target)
    matched_pred: set[int] = set()
    true_positive = 0
    lesion_dice = []
    for target_id in range(1, target_count + 1):
        target_component = target_labels == target_id
        best = (0.0, 0)
        for pred_id in range(1, pred_count + 1):
            pred_component = pred_labels == pred_id
            overlap = np.logical_and(target_component, pred_component).sum()
            score = overlap / max(1, target_component.sum())
            if score > best[0]:
                best = (float(score), pred_id)
        if best[0] >= overlap_threshold and best[1] not in matched_pred:
            true_positive += 1
            matched_pred.add(best[1])
            lesion_dice.append(dice_fg(pred_labels == best[1], target_component))
    precision = true_positive / pred_count if pred_count else float(target_count == 0)
    recall = true_positive / target_count if target_count else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "lesion_dice_mean": float(np.mean(lesion_dice)) if lesion_dice else 0.0,
        "lesion_precision": precision,
        "lesion_recall": recall,
        "lesion_f1": f1,
        "fp_per_scan": float(pred_count - true_positive),
        "absolute_lesion_count_difference": float(abs(pred_count - target_count)),
    }


def hd95_mm(
    prediction: np.ndarray, target: np.ndarray, spacing: tuple[float, float, float]
) -> float:
    prediction, target = prediction.astype(bool), target.astype(bool)
    if not prediction.any() or not target.any():
        return float("nan")
    pred_surface = prediction ^ ndimage.binary_erosion(prediction)
    target_surface = target ^ ndimage.binary_erosion(target)
    to_target = ndimage.distance_transform_edt(~target_surface, sampling=spacing)[pred_surface]
    to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)[target_surface]
    return float(np.percentile(np.concatenate([to_target, to_pred]), 95))


def case_metrics(
    prediction: np.ndarray, target: np.ndarray, spacing=(1.0, 1.0, 1.0)
) -> dict[str, float]:
    voxel_ml = float(np.prod(spacing) / 1000)
    result = lesion_metrics(prediction, target)
    result.update(
        {
            "dice_fg_case_mean": dice_fg(prediction, target) if target.any() else float("nan"),
            "dice_fg_global": dice_fg(prediction, target),
            "hd95_mm": hd95_mm(prediction, target, spacing),
            "nsd": float("nan"),
            "avd_ml": abs(float(prediction.sum() - target.sum())) * voxel_ml,
            "rvd_percent": float((prediction.sum() - target.sum()) / target.sum() * 100)
            if target.any()
            else float("nan"),
            "froc_score": result["lesion_recall"] if result["fp_per_scan"] <= 2 else 0.0,
            "patient_specificity": float(not prediction.any())
            if not target.any()
            else float("nan"),
            "false_positive_volume_ml": float(prediction.sum()) * voxel_ml
            if not target.any()
            else 0.0,
        }
    )
    return result
