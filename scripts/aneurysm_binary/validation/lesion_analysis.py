#!/usr/bin/env python3
"""Lesion-level analysis for an nnU-Net full-volume validation directory."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage


SIZE_LABELS = [
    "<0.01 mL",
    "0.01–0.05 mL",
    "0.05–0.1 mL",
    "0.1–0.5 mL",
    "≥0.5 mL",
]


def analyze_lesions(
    summary_path: Path, prediction_dir: Path, output_dir: Path
) -> dict[str, float | int | str]:
    """Match each GT component to any overlapping prediction component."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    structure = ndimage.generate_binary_structure(rank=3, connectivity=3)
    lesion_rows: list[dict] = []

    for case_item in summary.get("metric_per_case", []):
        reference_path = Path(case_item["reference_file"])
        prediction_path = prediction_dir / Path(case_item["prediction_file"]).name
        if not prediction_path.is_file():
            raise FileNotFoundError(f"예측 파일 없음: {prediction_path}")

        gt_image = nib.load(reference_path)
        pred_image = nib.load(prediction_path)
        gt = np.asanyarray(gt_image.dataobj) > 0
        pred = np.asanyarray(pred_image.dataobj) > 0
        if gt.shape != pred.shape:
            raise ValueError(
                f"shape 불일치: {reference_path.name}: GT={gt.shape}, pred={pred.shape}"
            )

        voxel_volume_ml = abs(float(np.linalg.det(gt_image.affine[:3, :3]))) / 1000.0
        gt_labels, _ = ndimage.label(gt, structure=structure)
        pred_labels, _ = ndimage.label(pred, structure=structure)

        for lesion_id, lesion_slice in enumerate(
            ndimage.find_objects(gt_labels), start=1
        ):
            if lesion_slice is None:
                continue
            gt_component = gt_labels[lesion_slice] == lesion_id
            overlapping_pred_ids = np.unique(
                pred_labels[lesion_slice][gt_component]
            )
            overlapping_pred_ids = overlapping_pred_ids[overlapping_pred_ids > 0]
            detected = len(overlapping_pred_ids) > 0
            if detected:
                pred_component = np.isin(pred_labels, overlapping_pred_ids)
                intersection = int(
                    np.count_nonzero(pred_component & (gt_labels == lesion_id))
                )
                denominator = int(pred_component.sum()) + int(gt_component.sum())
                lesion_dice = 2.0 * intersection / denominator if denominator else 0.0
            else:
                intersection = 0
                lesion_dice = 0.0

            lesion_rows.append(
                {
                    "case_id": prediction_path.name.removesuffix(".nii.gz"),
                    "lesion_id": lesion_id,
                    "gt_voxels": int(gt_component.sum()),
                    "gt_volume_ml": float(gt_component.sum() * voxel_volume_ml),
                    "detected": bool(detected),
                    "overlap_voxels": intersection,
                    "lesion_dice": float(lesion_dice),
                }
            )
        del gt, pred, gt_labels, pred_labels
        gc.collect()

    lesion_df = pd.DataFrame(lesion_rows)
    if lesion_df.empty:
        raise RuntimeError("GT 병변 component가 없습니다.")
    lesion_df.to_csv(output_dir / "lesion_size_metrics.csv", index=False)

    lesion_df["size_group"] = pd.cut(
        lesion_df["gt_volume_ml"],
        bins=[-np.inf, 0.01, 0.05, 0.1, 0.5, np.inf],
        labels=SIZE_LABELS,
        right=False,
    )
    size_summary = (
        lesion_df.groupby("size_group", observed=False)
        .agg(
            gt_lesions=("lesion_id", "size"),
            detected_lesions=("detected", "sum"),
            detection_rate=("detected", "mean"),
            mean_lesion_dice=("lesion_dice", "mean"),
            median_lesion_dice=("lesion_dice", "median"),
            mean_dice_detected_only=(
                "lesion_dice",
                lambda values: values[
                    lesion_df.loc[values.index, "detected"]
                ].mean(),
            ),
            median_volume_ml=("gt_volume_ml", "median"),
        )
        .reset_index()
    )
    size_summary["detected/total"] = (
        size_summary["detected_lesions"].astype(int).astype(str)
        + "/"
        + size_summary["gt_lesions"].astype(int).astype(str)
    )
    size_summary.to_csv(output_dir / "lesion_size_summary.csv", index=False)

    detected = int(lesion_df["detected"].sum())
    total = int(len(lesion_df))
    detected_dice = lesion_df.loc[lesion_df["detected"], "lesion_dice"]
    result: dict[str, float | int | str] = {
        "total_lesions": total,
        "detected_lesions": detected,
        "lesion_detection_rate": float(detected / total),
        "mean_lesion_dice": float(lesion_df["lesion_dice"].mean()),
        "mean_dice_detected_only": float(detected_dice.mean()),
        "metrics_csv": str(output_dir / "lesion_size_metrics.csv"),
        "size_summary_csv": str(output_dir / "lesion_size_summary.csv"),
    }
    (output_dir / "lesion_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--prediction-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = analyze_lesions(args.summary, args.prediction_dir, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
