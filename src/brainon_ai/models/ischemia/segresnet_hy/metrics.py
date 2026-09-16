"""Metrics preserved from the validated I-S00 evaluator."""

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt, label as cc_label


def binary_dice(logits, target, threshold=0.5):
    prediction = (torch.sigmoid(logits) >= threshold).float()
    target = (target > 0.5).float()

    intersection = (prediction * target).sum()
    denominator = prediction.sum() + target.sum()

    if denominator == 0:
        return 1.0

    return float((2.0 * intersection / denominator).item())


def calculate_voxel_metrics(pred, gt):
    """
    pred, gt : binary numpy array

    모든 지표는 환자 1명 단위로 계산.
    """

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())

    pred_positive = tp + fp
    gt_positive = tp + fn

    # --------------------------------------------------------
    # Dice
    # 학습 코드의 binary_dice와 동일한 empty-mask 정책
    # --------------------------------------------------------

    dice_den = 2 * tp + fp + fn

    if dice_den == 0:
        dice = 1.0
    else:
        dice = (2.0 * tp) / dice_den

    # --------------------------------------------------------
    # IoU / Jaccard
    # --------------------------------------------------------

    iou_den = tp + fp + fn

    if iou_den == 0:
        iou = 1.0
    else:
        iou = tp / iou_den

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    if pred_positive == 0:
        precision = 1.0 if gt_positive == 0 else 0.0
    else:
        precision = tp / pred_positive

    # --------------------------------------------------------
    # Recall / Sensitivity
    # --------------------------------------------------------

    if gt_positive == 0:
        recall = 1.0 if pred_positive == 0 else 0.0
    else:
        recall = tp / gt_positive

    # --------------------------------------------------------
    # Specificity
    # --------------------------------------------------------

    specificity_den = tn + fp

    if specificity_den == 0:
        specificity = 1.0
    else:
        specificity = tn / specificity_den

    # --------------------------------------------------------
    # NPV
    # --------------------------------------------------------

    npv_den = tn + fn

    if npv_den == 0:
        npv = 1.0
    else:
        npv = tn / npv_den

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    total = tp + fp + fn + tn

    accuracy = (
        (tp + tn) / total
        if total > 0
        else 1.0
    )

    # --------------------------------------------------------
    # Balanced Accuracy
    # --------------------------------------------------------

    balanced_accuracy = (
        recall + specificity
    ) / 2.0

    # --------------------------------------------------------
    # F1
    # Binary segmentation에서는 Dice와 동일한 식
    # --------------------------------------------------------

    f1 = dice

    # --------------------------------------------------------
    # Matthews Correlation Coefficient
    # --------------------------------------------------------

    mcc_den = np.sqrt(
    float(tp + fp)
    * float(tp + fn)
    * float(tn + fp)
    * float(tn + fn)
)

    if mcc_den == 0:
        if gt_positive == 0 and pred_positive == 0:
            mcc = 1.0
        else:
            mcc = 0.0
    else:
        mcc = (
            (tp * tn) - (fp * fn)
        ) / mcc_den

    return {
        "tp_voxels": tp,
        "fp_voxels": fp,
        "fn_voxels": fn,
        "tn_voxels": tn,

        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "npv": npv,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "mcc": mcc,
    }


# ============================================================
# Surface-distance metrics
# ============================================================


def get_surface(mask):
    """
    Binary mask의 surface voxel 추출.
    """

    mask = mask.astype(bool)

    if not mask.any():
        return np.zeros_like(mask, dtype=bool)

    structure = np.ones(
        (3, 3, 3),
        dtype=bool,
    )

    eroded = binary_erosion(
        mask,
        structure=structure,
        border_value=0,
    )

    return np.logical_and(
        mask,
        ~eroded,
    )


def calculate_surface_metrics(
    pred,
    gt,
    spacing,
):
    """
    HD95 / Hausdorff Max / ASSD
    단위 = mm
    """

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    pred_exists = pred.any()
    gt_exists = gt.any()

    # 둘 다 병변 없음
    if not pred_exists and not gt_exists:
        return {
            "hd95_mm": 0.0,
            "hausdorff_max_mm": 0.0,
            "assd_mm": 0.0,
        }

    # 한쪽만 병변 존재
    # surface distance 정의 불가능
    if pred_exists != gt_exists:
        return {
            "hd95_mm": np.nan,
            "hausdorff_max_mm": np.nan,
            "assd_mm": np.nan,
        }

    pred_surface = get_surface(pred)
    gt_surface = get_surface(gt)

    # target surface까지 거리
    distance_to_gt = distance_transform_edt(
        ~gt_surface,
        sampling=spacing,
    )

    distance_to_pred = distance_transform_edt(
        ~pred_surface,
        sampling=spacing,
    )

    pred_to_gt = distance_to_gt[
        pred_surface
    ]

    gt_to_pred = distance_to_pred[
        gt_surface
    ]

    # Hausdorff 95
    hd95 = max(
        np.percentile(pred_to_gt, 95),
        np.percentile(gt_to_pred, 95),
    )

    # Maximum Hausdorff
    hd_max = max(
        pred_to_gt.max(),
        gt_to_pred.max(),
    )

    # Average Symmetric Surface Distance
    all_distances = np.concatenate(
        [
            pred_to_gt,
            gt_to_pred,
        ]
    )

    assd = all_distances.mean()

    return {
        "hd95_mm": float(hd95),
        "hausdorff_max_mm": float(hd_max),
        "assd_mm": float(assd),
    }


# ============================================================
# Volume metrics
# ============================================================


def calculate_volume_metrics(
    pred,
    gt,
    spacing,
):
    voxel_volume_mm3 = float(
        np.prod(spacing)
    )

    gt_voxels = int(gt.sum())
    pred_voxels = int(pred.sum())

    gt_volume_ml = (
        gt_voxels
        * voxel_volume_mm3
        / 1000.0
    )

    pred_volume_ml = (
        pred_voxels
        * voxel_volume_mm3
        / 1000.0
    )

    abs_volume_error_ml = abs(
        pred_volume_ml
        - gt_volume_ml
    )

    if gt_volume_ml > 0:
        relative_volume_error_pct = (
            (
                pred_volume_ml
                - gt_volume_ml
            )
            / gt_volume_ml
            * 100.0
        )
    else:
        relative_volume_error_pct = (
            0.0
            if pred_volume_ml == 0
            else np.nan
        )

    volume_sum = (
        gt_volume_ml
        + pred_volume_ml
    )

    if volume_sum == 0:
        volume_similarity = 1.0
    else:
        volume_similarity = (
            1.0
            - abs(
                pred_volume_ml
                - gt_volume_ml
            )
            / volume_sum
        )

    return {
        "voxel_volume_mm3": voxel_volume_mm3,

        "gt_volume_ml": gt_volume_ml,
        "pred_volume_ml": pred_volume_ml,

        "abs_volume_error_ml":
            abs_volume_error_ml,

        "relative_volume_error_pct":
            relative_volume_error_pct,

        "volume_similarity":
            volume_similarity,
    }


# ============================================================
# Lesion-level metrics
# ============================================================


def calculate_lesion_metrics(
    pred,
    gt,
):
    """
    26-connectivity 기준 connected component.

    GT lesion이 prediction과 voxel 1개 이상 겹치면
    detected lesion으로 판정.
    """

    structure = np.ones(
        (3, 3, 3),
        dtype=int,
    )

    gt_cc, gt_lesions = cc_label(
        gt.astype(bool),
        structure=structure,
    )

    pred_cc, pred_lesions = cc_label(
        pred.astype(bool),
        structure=structure,
    )

    detected_gt = 0

    for lesion_id in range(
        1,
        gt_lesions + 1,
    ):
        lesion_mask = (
            gt_cc == lesion_id
        )

        if np.logical_and(
            lesion_mask,
            pred,
        ).any():
            detected_gt += 1

    matched_predictions = 0

    for lesion_id in range(
        1,
        pred_lesions + 1,
    ):
        lesion_mask = (
            pred_cc == lesion_id
        )

        if np.logical_and(
            lesion_mask,
            gt,
        ).any():
            matched_predictions += 1

    lesion_fn = (
        gt_lesions
        - detected_gt
    )

    lesion_fp = (
        pred_lesions
        - matched_predictions
    )

    # Lesion recall
    if gt_lesions == 0:
        lesion_recall = (
            1.0
            if pred_lesions == 0
            else 0.0
        )
    else:
        lesion_recall = (
            detected_gt
            / gt_lesions
        )

    # Lesion precision
    if pred_lesions == 0:
        lesion_precision = (
            1.0
            if gt_lesions == 0
            else 0.0
        )
    else:
        lesion_precision = (
            matched_predictions
            / pred_lesions
        )

    if (
        lesion_precision
        + lesion_recall
    ) == 0:
        lesion_f1 = 0.0
    else:
        lesion_f1 = (
            2
            * lesion_precision
            * lesion_recall
            / (
                lesion_precision
                + lesion_recall
            )
        )

    return {
        "gt_lesion_count":
            int(gt_lesions),

        "pred_lesion_count":
            int(pred_lesions),

        "detected_gt_lesions":
            int(detected_gt),

        "false_negative_lesions":
            int(lesion_fn),

        "false_positive_lesions":
            int(lesion_fp),

        "lesion_precision":
            lesion_precision,

        "lesion_recall":
            lesion_recall,

        "lesion_f1":
            lesion_f1,
    }


# ============================================================
# Bootstrap 95% CI
# ============================================================


def bootstrap_mean_ci(
    values,
    n_bootstrap=10000,
    seed=42,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan, np.nan

    if len(values) == 1:
        return values[0], values[0]

    rng = np.random.default_rng(seed)

    bootstrap_means = np.empty(
        n_bootstrap,
        dtype=float,
    )

    for i in range(n_bootstrap):
        sample = rng.choice(
            values,
            size=len(values),
            replace=True,
        )

        bootstrap_means[i] = (
            sample.mean()
        )

    lower = np.percentile(
        bootstrap_means,
        2.5,
    )

    upper = np.percentile(
        bootstrap_means,
        97.5,
    )

    return float(lower), float(upper)


# ============================================================
# 평가
# ============================================================
