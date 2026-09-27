#!/usr/bin/env python3
"""BrainCDSS Metric Standard v1.0 evaluator for binary 3-D segmentation masks.

This tool evaluates already exported *full-volume, original-space* NIfTI masks.
It deliberately does not perform patch inference: checkpoint inference belongs to
the model-specific pipeline, while this script makes the final metric contract
identical for CT, MRI, and MRA.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure, label
from scipy.spatial import cKDTree

CONNECTIVITY = generate_binary_structure(3, 3)
CT = "HEMORRHAGIC_STROKE_CT"
MRI = "ISCHEMIC_STROKE_DWI_ADC"
MRA = "CEREBRAL_ANEURYSM_MRA"


def value_or_null(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def case_id(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    for suffix in ("_mask", "_pred_mask", "_prediction"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    return label(mask, structure=CONNECTIVITY)


def surface_distances_mm(reference: np.ndarray, prediction: np.ndarray, spacing: tuple[float, ...]) -> tuple[float | None, float | None]:
    """Symmetric ASSD and HD95; undefined when either foreground is empty."""
    if not reference.any() or not prediction.any():
        return None, None
    ref_surface = reference & ~binary_erosion(reference, structure=CONNECTIVITY, border_value=0)
    pred_surface = prediction & ~binary_erosion(prediction, structure=CONNECTIVITY, border_value=0)
    ref_points = np.argwhere(ref_surface) * np.asarray(spacing)
    pred_points = np.argwhere(pred_surface) * np.asarray(spacing)
    if not len(ref_points) or not len(pred_points):
        return None, None
    forward = cKDTree(pred_points).query(ref_points, k=1)[0]
    backward = cKDTree(ref_points).query(pred_points, k=1)[0]
    distances = np.concatenate((forward, backward))
    return float(distances.mean()), float(np.percentile(distances, 95))


def evaluate(pairs: list[tuple[Path, Path]], analysis_type: str, split_name: str, model_id: str) -> dict:
    positive_dice: list[float] = []
    positive_iou: list[float] = []
    assd_values: list[float] = []
    hd95_values: list[float] = []
    small_dice: list[float] = []
    total_tp = total_fp_voxel = total_fn = 0
    total_gt_lesions = total_pred_lesions = total_false_positive_lesions = 0
    detected_lesions = positive_cases = negative_cases = detected_cases = negative_cases_with_fp = 0
    small_gt_lesions = small_detected = 0
    per_case: list[dict] = []

    for prediction_path, label_path in pairs:
        pred_img, gt_img = nib.load(str(prediction_path)), nib.load(str(label_path))
        pred, gt = np.asanyarray(pred_img.dataobj) > 0, np.asanyarray(gt_img.dataobj) > 0
        if pred.shape != gt.shape:
            raise ValueError(f"shape mismatch: {prediction_path} {pred.shape} != {label_path} {gt.shape}")
        spacing = tuple(float(x) for x in gt_img.header.get_zooms()[:3])
        voxel_ml = float(np.prod(spacing) / 1000.0)
        tp, fp, fn = int((pred & gt).sum()), int((pred & ~gt).sum()), int((~pred & gt).sum())
        total_tp += tp; total_fp_voxel += fp; total_fn += fn
        gt_cc, gt_n = components(gt); pred_cc, pred_n = components(pred)
        detected = sum(bool(pred[gt_cc == i].any()) for i in range(1, gt_n + 1))
        false_positive = sum(not bool(gt[pred_cc == i].any()) for i in range(1, pred_n + 1))
        total_gt_lesions += gt_n; total_pred_lesions += pred_n
        detected_lesions += detected; total_false_positive_lesions += false_positive
        assd, hd95 = surface_distances_mm(gt, pred, spacing)
        if assd is not None: assd_values.append(assd)
        if hd95 is not None: hd95_values.append(hd95)

        is_positive = bool(gt.any())
        if is_positive:
            positive_cases += 1
            denominator = 2 * tp + fp + fn
            union = tp + fp + fn
            positive_dice.append(2 * tp / denominator if denominator else 1.0)
            positive_iou.append(tp / union if union else 1.0)
            # A case counts as detected only when some prediction overlaps GT.
            detected_cases += int(bool((pred & gt).any()))
            if analysis_type != MRA:
                for component_id in range(1, gt_n + 1):
                    component = gt_cc == component_id
                    if component.sum() * voxel_ml < 5.0:
                        small_gt_lesions += 1
                        # Component-level Dice against the prediction component(s) that overlap it.
                        matched_ids = np.unique(pred_cc[component]); matched_ids = matched_ids[matched_ids > 0]
                        matched = np.isin(pred_cc, matched_ids) if len(matched_ids) else np.zeros_like(pred)
                        overlap = int((matched & component).sum())
                        component_denom = int(matched.sum()) + int(component.sum())
                        small_dice.append(2 * overlap / component_denom if component_denom else 0.0)
                        small_detected += int(bool(matched_ids.size))
        else:
            negative_cases += 1
            negative_cases_with_fp += int(pred.any())
        per_case.append({
            "case_id": case_id(label_path), "positive": is_positive, "dice": (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else None,
            "iou": (tp / (tp + fp + fn)) if (tp + fp + fn) else None, "assd_mm": assd, "hd95_mm": hd95,
            "gt_lesions": gt_n, "pred_lesions": pred_n, "detected_lesions": detected, "false_positive_lesions": false_positive,
        })

    lesion_recall = detected_lesions / total_gt_lesions if total_gt_lesions else None
    matched_predictions = total_pred_lesions - total_false_positive_lesions
    lesion_precision = matched_predictions / total_pred_lesions if total_pred_lesions else None
    lesion_f1 = (2 * lesion_precision * lesion_recall / (lesion_precision + lesion_recall)
                 if lesion_precision is not None and lesion_recall is not None and lesion_precision + lesion_recall else None)
    result = {
        "metric_standard": "BrainCDSS Final Model Metric Standard v1.0",
        "analysis_type": analysis_type, "model_id": model_id, "evaluation_split": split_name,
        "full_volume_original_space": True,
        "warning": None if split_name == "test" else "This is not a final operating metric: evaluation_split is not test.",
        "test_macro_dice_positive": value_or_null(positive_dice),
        "test_macro_iou_positive": value_or_null(positive_iou),
        "test_voxel_recall_positive": total_tp / (total_tp + total_fn) if total_tp + total_fn else None,
        "test_voxel_precision_positive": total_tp / (total_tp + total_fp_voxel) if total_tp + total_fp_voxel else None,
        "test_case_detection_rate": detected_cases / positive_cases if positive_cases else None,
        "test_negative_case_fp_rate": negative_cases_with_fp / negative_cases if negative_cases else None,
        "test_lesion_recall": lesion_recall, "test_lesion_precision": lesion_precision, "test_lesion_f1": lesion_f1,
        "test_fp_per_case": total_false_positive_lesions / len(pairs) if pairs else None,
        "test_small_lesion_dice": value_or_null(small_dice) if analysis_type != MRA else None,
        "test_small_lesion_recall": small_detected / small_gt_lesions if small_gt_lesions else None if analysis_type != MRA else None,
        "test_mean_assd_mm": value_or_null(assd_values), "test_hd95_mm": value_or_null(hd95_values),
        "total_cases": len(pairs), "positive_cases": positive_cases, "negative_cases": negative_cases,
        "total_gt_lesions": total_gt_lesions, "total_pred_lesions": total_pred_lesions,
        "small_gt_lesions": small_gt_lesions if analysis_type != MRA else None,
        "metric_definition": {"small_lesion_threshold_ml": 5.0 if analysis_type != MRA else None,
                              "lesion_match": "3-D connected component has at least one overlapping voxel"},
        "per_case": per_case,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-type", choices=[CT, MRI, MRA], required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--prediction-template", default="{case_id}.nii.gz", help="e.g. {case_id}_t0.25.nii.gz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-list", type=Path, help="Optional text file: one case ID per line; enforces a fixed cohort.")
    args = parser.parse_args()
    labels = sorted(list(args.label_dir.glob("*.nii.gz")) + list(args.label_dir.glob("*.nii")))
    if args.case_list:
        wanted = {line.strip() for line in args.case_list.read_text(encoding="utf-8").splitlines() if line.strip()}
        labels = [path for path in labels if case_id(path) in wanted]
        if not labels:
            raise ValueError(f"No labels matched case list: {args.case_list}")
    if not labels: raise FileNotFoundError(f"No NIfTI labels: {args.label_dir}")
    pairs = []
    missing = []
    for label_path in labels:
        cid = case_id(label_path)
        prediction_path = args.prediction_dir / args.prediction_template.format(case_id=cid)
        if prediction_path.is_file():
            pairs.append((prediction_path, label_path))
        else:
            missing.append(str(prediction_path))
    if missing: raise FileNotFoundError("Missing prediction(s), first five:\n" + "\n".join(missing[:5]))
    result = evaluate(pairs, args.analysis_type, args.split_name, args.model_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in {"per_case", "metric_definition"}}, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
