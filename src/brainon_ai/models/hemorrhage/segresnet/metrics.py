from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch


def binary_case_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float | int]:
    """한 CT의 foreground(출혈=1) confusion count와 지표를 계산합니다."""
    pred = prediction.detach().to(device="cpu", dtype=torch.bool).reshape(-1)
    truth = target.detach().to(device="cpu", dtype=torch.bool).reshape(-1)
    tp = int(torch.count_nonzero(pred & truth))
    fp = int(torch.count_nonzero(pred & ~truth))
    fn = int(torch.count_nonzero(~pred & truth))
    pred_voxels = tp + fp
    target_voxels = tp + fn

    if target_voxels == 0:
        dice = 1.0 if pred_voxels == 0 else 0.0
        recall = float("nan")
        precision = 1.0 if pred_voxels == 0 else 0.0
    else:
        dice = 2.0 * tp / max(2 * tp + fp + fn, 1)
        recall = tp / target_voxels
        precision = tp / pred_voxels if pred_voxels > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predicted_voxels": pred_voxels,
        "target_voxels": target_voxels,
        "target_positive": int(target_voxels > 0),
        "dice": float(dice),
        "recall": float(recall),
        "precision": float(precision),
    }


def _finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else float("nan")


def _source_summary(frame: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for source, subset in frame.groupby("source", sort=True):
        positive = subset.loc[subset.target_positive.eq(1)]
        empty = subset.loc[subset.target_positive.eq(0)]
        result[str(source)] = {
            "cases": int(len(subset)),
            "positive_cases": int(len(positive)),
            "dice_positive_macro": _finite_mean(positive["dice"]),
            "recall_positive_macro": _finite_mean(positive["recall"]),
            "empty_gt_false_positive_rate": _finite_mean(empty["predicted_voxels"].gt(0).astype(float)),
        }
    return result


def summarize_cases(rows: list[dict], dice_weight: float, recall_weight: float) -> dict:
    if not rows:
        raise ValueError("평가할 사례가 없습니다.")
    frame = pd.DataFrame(rows)
    positive = frame.loc[frame.target_positive.eq(1)]
    empty = frame.loc[frame.target_positive.eq(0)]
    if positive.empty:
        raise RuntimeError("양성 GT 사례가 없어 Dice/Recall 기준을 계산할 수 없습니다.")

    dice = _finite_mean(positive["dice"])
    recall = _finite_mean(positive["recall"])
    tp = int(frame["tp"].sum())
    fp = int(frame["fp"].sum())
    fn = int(frame["fn"].sum())
    composite = float(dice_weight) * dice + float(recall_weight) * recall

    return {
        "cases": int(len(frame)),
        "positive_cases": int(len(positive)),
        "empty_gt_cases": int(len(empty)),
        "dice_positive_macro": dice,
        "recall_positive_macro": recall,
        "precision_positive_macro": _finite_mean(positive["precision"]),
        "dice_all_cases_macro": _finite_mean(frame["dice"]),
        "foreground_micro_dice": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "empty_gt_false_positive_rate": _finite_mean(empty["predicted_voxels"].gt(0).astype(float)),
        "composite": composite,
        "dice_weight": float(dice_weight),
        "recall_weight": float(recall_weight),
        "source_metrics": _source_summary(frame),
    }
