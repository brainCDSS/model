from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch


def confusion_by_class(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 53,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = (
        prediction.detach()
        .to(device="cpu", dtype=torch.int64)
        .reshape(-1)
    )
    truth = (
        target.detach()
        .to(device="cpu", dtype=torch.int64)
        .reshape(-1)
    )

    if pred.numel() != truth.numel():
        raise ValueError(
            "Prediction and target voxel counts differ"
        )

    if pred.numel() and (
        int(pred.min()) < 0
        or int(pred.max()) >= num_classes
        or int(truth.min()) < 0
        or int(truth.max()) >= num_classes
    ):
        raise ValueError(
            "Prediction or target contains an out-of-range class"
        )

    encoded = truth * num_classes + pred

    matrix = torch.bincount(
        encoded,
        minlength=num_classes**2,
    ).reshape(num_classes, num_classes)

    diagonal = matrix.diag()

    tp = diagonal.numpy().astype(
        np.int64,
        copy=False,
    )
    fp = (
        matrix.sum(dim=0) - diagonal
    ).numpy().astype(np.int64, copy=False)
    fn = (
        matrix.sum(dim=1) - diagonal
    ).numpy().astype(np.int64, copy=False)

    return tp, fp, fn


def foreground_dice(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
) -> float:
    denominator = (
        2 * tp[1:] + fp[1:] + fn[1:]
    )

    scores = np.full(
        denominator.shape,
        np.nan,
        dtype=np.float64,
    )

    valid = denominator > 0
    scores[valid] = (
        2.0 * tp[1:][valid] / denominator[valid]
    )

    if not np.any(valid):
        return float("nan")

    return float(np.nanmean(scores))


def patient_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 53,
) -> tuple[
    dict,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    tp, fp, fn = confusion_by_class(
        prediction,
        target,
        num_classes,
    )

    pred_fg = (
        prediction.detach()
        .to(device="cpu")
        .ne(0)
    )
    truth_fg = (
        target.detach()
        .to(device="cpu")
        .ne(0)
    )

    merged_tp = int(
        torch.count_nonzero(pred_fg & truth_fg)
    )
    merged_fp = int(
        torch.count_nonzero(pred_fg & ~truth_fg)
    )
    merged_fn = int(
        torch.count_nonzero(~pred_fg & truth_fg)
    )

    predicted_voxels = merged_tp + merged_fp
    target_voxels = merged_tp + merged_fn

    if target_voxels == 0:
        patient_dice = (
            float("nan")
            if predicted_voxels == 0
            else 0.0
        )
        precision = (
            float("nan")
            if predicted_voxels == 0
            else 0.0
        )
        recall = float("nan")
    else:
        denominator = (
            2 * merged_tp + merged_fp + merged_fn
        )
        patient_dice = (
            2.0 * merged_tp / denominator
        )
        precision = (
            merged_tp / predicted_voxels
            if predicted_voxels
            else 0.0
        )
        recall = merged_tp / target_voxels

    present_classes = (
        np.flatnonzero(
            (tp[1:] + fn[1:]) > 0
        )
        + 1
    )

    class_dice = []

    for class_id in present_classes:
        denominator = (
            2 * tp[class_id]
            + fp[class_id]
            + fn[class_id]
        )
        class_dice.append(
            2.0 * tp[class_id] / denominator
        )

    patient_macro = (
        float(np.mean(class_dice))
        if class_dice
        else float("nan")
    )

    row = {
        "patient_dice": float(patient_dice),
        "patient_macro_dice": patient_macro,
        "patient_macro_gt_present_classes": ";".join(
            str(value)
            for value in present_classes
        ),
        "precision": float(precision),
        "recall": float(recall),
        "target_positive": int(target_voxels > 0),
        "predicted_positive": int(predicted_voxels > 0),
        "false_positive_negative_case": int(
            target_voxels == 0
            and predicted_voxels > 0
        ),
        "tp": merged_tp,
        "fp": merged_fp,
        "fn": merged_fn,
    }

    return row, tp, fp, fn


def _finite_mean(values) -> float:
    array = np.asarray(
        list(values),
        dtype=np.float64,
    )
    finite = array[np.isfinite(array)]

    return (
        float(finite.mean())
        if finite.size
        else float("nan")
    )


def _finite_median(values) -> float:
    array = np.asarray(
        list(values),
        dtype=np.float64,
    )
    finite = array[np.isfinite(array)]

    return (
        float(np.median(finite))
        if finite.size
        else float("nan")
    )


def summarize_final_validation(
    rows: list[dict],
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    if not rows:
        raise ValueError(
            "Final validation contains no completed patients"
        )

    frame = pd.DataFrame(rows)
    class_rows = []

    for class_id in range(1, len(tp)):
        denominator = (
            2 * tp[class_id]
            + fp[class_id]
            + fn[class_id]
        )

        gt_present = bool(
            tp[class_id] + fn[class_id] > 0
        )

        dice = (
            2 * tp[class_id] / denominator
            if denominator
            else float("nan")
        )

        predicted = (
            tp[class_id] + fp[class_id]
        )

        precision = (
            tp[class_id] / predicted
            if predicted
            else (
                0.0
                if gt_present
                else float("nan")
            )
        )

        recall = (
            tp[class_id]
            / (tp[class_id] + fn[class_id])
            if gt_present
            else float("nan")
        )

        class_rows.append(
            {
                "class_id": class_id,
                "TP": int(tp[class_id]),
                "FP": int(fp[class_id]),
                "FN": int(fn[class_id]),
                "dice": float(dice),
                "precision": float(precision),
                "recall": float(recall),
                "gt_present": gt_present,
            }
        )

    per_class = pd.DataFrame(class_rows)
    gt_present = per_class.loc[
        per_class["gt_present"]
    ]
    negative = frame.loc[
        frame["target_positive"].eq(0)
    ]

    summary = {
        "validation_cases": int(len(frame)),
        "positive_cases": int(
            frame["target_positive"].sum()
        ),
        "negative_cases": int(len(negative)),
        "val_patient_mean_dice": _finite_mean(
            frame["patient_dice"]
        ),
        "val_patient_median_dice": _finite_median(
            frame["patient_dice"]
        ),
        "val_patient_mean_macro_dice": _finite_mean(
            frame["patient_macro_dice"]
        ),
        "val_patient_mean_precision": _finite_mean(
            frame["precision"]
        ),
        "val_patient_mean_recall": _finite_mean(
            frame["recall"]
        ),
        "val_macro_dice": _finite_mean(
            gt_present["dice"]
        ),
        "val_macro_precision": _finite_mean(
            gt_present["precision"]
        ),
        "val_macro_recall": _finite_mean(
            gt_present["recall"]
        ),
        "negative_case_fp_rate": (
            float(
                negative[
                    "false_positive_negative_case"
                ].mean()
            )
            if len(negative)
            else float("nan")
        ),
    }

    required_metrics = (
        "val_patient_mean_dice",
        "val_macro_dice",
        "val_macro_recall",
    )

    if not all(
        math.isfinite(summary[key])
        for key in required_metrics
    ):
        raise RuntimeError(
            "Final validation produced non-finite "
            "required aggregate metrics"
        )

    return summary, per_class
