#!/usr/bin/env python3
"""Lesion-level rescue/miss analysis: I-N120 versus E03 ensemble."""
from __future__ import annotations
import argparse, csv, json
from itertools import permutations
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import label

STRUCTURE = np.ones((3, 3, 3), dtype=np.uint8)
BINS = [0.01, 0.05, 0.1, 0.5]
LABELS = ["<0.01 mL", "0.01–0.05 mL", "0.05–0.1 mL", "0.1–0.5 mL", "≥0.5 mL"]

def components(mask): return label(mask, structure=STRUCTURE)
def n120_probability(path):
    with np.load(path) as archive:
        values = archive["probabilities"] if "probabilities" in archive else archive["softmax"]
    return np.asarray(values[-1], dtype=np.float32)
def align(probability, reference_shape, prediction_path):
    if probability.shape == reference_shape: return probability
    candidates = [probability.transpose(axis) for axis in permutations(range(3)) if probability.transpose(axis).shape == reference_shape]
    if not candidates: raise ValueError(f"Cannot align {probability.shape} to {reference_shape}")
    saved = np.asanyarray(nib.load(prediction_path).dataobj) > 0
    agreement = [np.mean((candidate >= .5) == saved) for candidate in candidates]
    return candidates[int(np.argmax(agreement))]
def detected_components(prediction, gt_components, count):
    return {idx: bool(prediction[gt_components == idx].any()) for idx in range(1, count + 1)}
def false_positive_count(prediction, gt):
    components_pred, count = components(prediction)
    return sum(not bool(gt[components_pred == idx].any()) for idx in range(1, count + 1))
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n120-probabilities", required=True, type=Path)
    ap.add_argument("--e03-predictions", required=True, type=Path)
    ap.add_argument("--label-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--n120-threshold", type=float, default=.15)
    ap.add_argument("--e03-threshold", type=float, default=.20)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, case_rows = [], []
    for probability_path in sorted(args.n120_probabilities.glob("*.npz")):
        case_id = probability_path.stem
        reference_path = args.label_dir / f"{case_id}.nii.gz"
        ensemble_path = args.e03_predictions / f"{case_id}.nii.gz"
        n120_prediction_path = args.n120_probabilities / f"{case_id}.nii.gz"
        if not reference_path.is_file() or not ensemble_path.is_file() or not n120_prediction_path.is_file(): continue
        reference = nib.load(reference_path); gt = np.asanyarray(reference.dataobj) > 0
        n120 = align(n120_probability(probability_path), gt.shape, n120_prediction_path) >= args.n120_threshold
        e03 = np.asanyarray(nib.load(ensemble_path).dataobj) > 0
        if e03.shape != gt.shape: raise ValueError(f"{case_id}: E03 {e03.shape} != GT {gt.shape}")
        gt_components, gt_count = components(gt)
        n120_detected = detected_components(n120, gt_components, gt_count)
        e03_detected = detected_components(e03, gt_components, gt_count)
        voxel_ml = float(np.prod(reference.header.get_zooms()[:3])) / 1000.0
        for lesion_id in range(1, gt_count + 1):
            volume_ml = int((gt_components == lesion_id).sum()) * voxel_ml
            before, after = n120_detected[lesion_id], e03_detected[lesion_id]
            status = "both_detected" if before and after else "e03_rescue" if after else "e03_regression" if before else "missed_by_both"
            rows.append({"case_id": case_id, "lesion_id": lesion_id, "gt_volume_ml": volume_ml, "n120_detected": before, "e03_detected": after, "status": status})
        case_rows.append({"case_id":case_id,"gt_lesions":gt_count,"n120_detected":sum(n120_detected.values()),"e03_detected":sum(e03_detected.values()),"n120_fp_lesions":false_positive_count(n120,gt),"e03_fp_lesions":false_positive_count(e03,gt)})
    lesions = pd.DataFrame(rows)
    lesions["size_group"] = pd.cut(lesions["gt_volume_ml"], [-np.inf,*BINS,np.inf], labels=LABELS, right=False)
    lesions.to_csv(args.output_dir / "lesion_rescue_detail.csv", index=False)
    cases = pd.DataFrame(case_rows); cases.to_csv(args.output_dir / "case_fp_comparison.csv", index=False)
    summary = lesions.groupby(["size_group", "status"], observed=False).size().unstack(fill_value=0).reindex(columns=["both_detected","e03_rescue","e03_regression","missed_by_both"],fill_value=0)
    summary["gt_lesions"] = summary.sum(axis=1)
    summary["n120_detection_rate"] = (summary["both_detected"] + summary["e03_regression"]) / summary["gt_lesions"]
    summary["e03_detection_rate"] = (summary["both_detected"] + summary["e03_rescue"]) / summary["gt_lesions"]
    summary = summary.reset_index(); summary.to_csv(args.output_dir / "rescue_by_size.csv", index=False)
    report = {"n120_threshold":args.n120_threshold,"e03_threshold":args.e03_threshold,"gt_lesions":len(lesions),"status_counts":lesions["status"].value_counts().to_dict(),"n120_fp_lesions":int(cases["n120_fp_lesions"].sum()),"e03_fp_lesions":int(cases["e03_fp_lesions"].sum()),"n120_fp_per_scan":float(cases["n120_fp_lesions"].mean()),"e03_fp_per_scan":float(cases["e03_fp_lesions"].mean())}
    (args.output_dir / "rescue_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
if __name__ == "__main__": raise SystemExit(main())
