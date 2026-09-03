from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader, Dataset

from brainon_ai.models.ischemia.segresnet.checkpointing import (
    load_checkpoint,
    load_model_state,
)
from brainon_ai.models.ischemia.segresnet.config import validate_config
from brainon_ai.models.ischemia.segresnet.data import (
    build_records,
    test_transform,
)
from brainon_ai.models.ischemia.segresnet.inference import infer_binary
from brainon_ai.models.ischemia.segresnet.metrics import (
    bootstrap_mean_ci,
    calculate_lesion_metrics,
    calculate_surface_metrics,
    calculate_volume_metrics,
    calculate_voxel_metrics,
)
from brainon_ai.models.ischemia.segresnet.network import build_model


METRIC_COLUMNS = [
    "dice", "iou", "precision", "recall_sensitivity", "specificity",
    "npv", "accuracy", "balanced_accuracy", "f1", "mcc",
    "hd95_mm", "hausdorff_max_mm", "assd_mm", "gt_volume_ml",
    "pred_volume_ml", "abs_volume_error_ml", "relative_volume_error_pct",
    "volume_similarity", "lesion_precision", "lesion_recall", "lesion_f1",
]


def _first(value):
    return value[0] if isinstance(value, (list, tuple)) else str(value)


def _case_status(pred, gt):
    gt_exists = bool(gt.any())
    pred_exists = bool(pred.any())
    if gt_exists and pred_exists:
        return "GT_POS_PRED_POS"
    if gt_exists and not pred_exists:
        return "GT_POS_PRED_EMPTY"
    if not gt_exists and pred_exists:
        return "GT_EMPTY_PRED_POS"
    return "GT_EMPTY_PRED_EMPTY"


def _summary(patient_df):
    rows = []
    for metric in METRIC_COLUMNS:
        values = pd.to_numeric(patient_df[metric], errors="coerce").to_numpy(dtype=float)
        valid = values[np.isfinite(values)]
        if len(valid) == 0:
            continue
        ci_low, ci_high = bootstrap_mean_ci(valid, n_bootstrap=10000, seed=42)
        rows.append(
            {
                "metric": metric,
                "n_valid": len(valid),
                "mean": float(np.mean(valid)),
                "std": float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0,
                "median": float(np.median(valid)),
                "q1": float(np.percentile(valid, 25)),
                "q3": float(np.percentile(valid, 75)),
                "min": float(np.min(valid)),
                "max": float(np.max(valid)),
                "mean_95ci_low": ci_low,
                "mean_95ci_high": ci_high,
            }
        )
    return pd.DataFrame(rows)


@torch.no_grad()
def evaluate_split(model, loader, device, config, prediction_dir=None):
    model.eval()
    patch_size = tuple(config["data"]["patch_size"])
    threshold = float(config["evaluation"]["threshold"])
    sw_batch_size = int(config["evaluation"]["sw_batch_size"])
    overlap = float(config["evaluation"]["overlap"])
    amp = bool(config["runtime"]["amp"])
    patient_results = []

    for index, batch in enumerate(loader, start=1):
        case_id = _first(batch["case_id"])
        label_path = _first(batch["label_path"])
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        prediction = infer_binary(
            model, image, patch_size, sw_batch_size, overlap, threshold, amp
        )
        pred_np = prediction[0, 0].cpu().numpy().astype(np.uint8)
        gt_np = (label[0, 0] > 0.5).cpu().numpy().astype(np.uint8)

        gt_nifti = nib.load(label_path)
        spacing = tuple(float(x) for x in gt_nifti.header.get_zooms()[:3])

        voxel_metrics = calculate_voxel_metrics(pred_np, gt_np)
        surface_metrics = calculate_surface_metrics(pred_np, gt_np, spacing)
        volume_metrics = calculate_volume_metrics(pred_np, gt_np, spacing)
        lesion_metrics = calculate_lesion_metrics(pred_np, gt_np)

        prediction_path = ""
        if prediction_dir is not None:
            pred_path = prediction_dir / f"{case_id}_pred_mask.nii.gz"
            header = gt_nifti.header.copy()
            header.set_data_dtype(np.uint8)
            nib.save(
                nib.Nifti1Image(pred_np, affine=gt_nifti.affine, header=header),
                pred_path,
            )
            prediction_path = str(pred_path)

        result = {
            "case_id": case_id,
            "case_status": _case_status(pred_np, gt_np),
            "spacing_x_mm": spacing[0],
            "spacing_y_mm": spacing[1],
            "spacing_z_mm": spacing[2],
            **voxel_metrics,
            **surface_metrics,
            **volume_metrics,
            **lesion_metrics,
            "prediction_path": prediction_path,
        }
        patient_results.append(result)
        hd95 = result["hd95_mm"]
        hd95_text = f"{hd95:.2f}" if np.isfinite(hd95) else "NA"
        print(
            f"[{index:02d}/{len(loader.dataset):02d}] {case_id} | "
            f"Dice={result['dice']:.4f} | IoU={result['iou']:.4f} | "
            f"Precision={result['precision']:.4f} | "
            f"Recall={result['recall_sensitivity']:.4f} | HD95={hd95_text}"
        )

    patient_df = pd.DataFrame(patient_results)
    return patient_df, _summary(patient_df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", required=True, choices=("val", "test"))
    parser.add_argument(
        "--output-dir",
        help="Optional new directory. Existing directories are rejected to prevent overwrites.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 사용할 수 없습니다.")
    device = torch.device("cuda")

    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, device)
    config = checkpoint["config"]
    validate_config(config)

    model = build_model(config).to(device)
    load_model_state(model, checkpoint)
    model.eval()

    records = build_records(Path(args.data_root), args.split, include_label_path=True)
    expected_cases = 38 if args.split == "val" else 37
    if len(records) != expected_cases:
        raise RuntimeError(
            f"{args.split} Case 수 오류: {len(records)} (예상 {expected_cases})"
        )
    loader = DataLoader(
        Dataset(records, transform=test_transform()),
        batch_size=1,
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=True,
    )

    output_dir = Path(args.output_dir) if args.output_dir else None
    prediction_dir = None
    if output_dir is not None:
        if output_dir.exists():
            raise FileExistsError(f"기존 평가 결과를 덮어쓸 수 없습니다: {output_dir}")
        prediction_dir = output_dir / f"{args.split}_predictions"
        prediction_dir.mkdir(parents=True)

    print("=" * 75)
    print(f"I-S00 SegResNet {args.split.upper()} Evaluation")
    print("=" * 75)
    print("Checkpoint       :", checkpoint_path)
    print("Best Epoch       :", int(checkpoint["epoch"]) + 1)
    print("Best Val Dice    :", float(checkpoint["best_dice"]))
    print("Cases            :", len(records))
    print("Threshold        :", config["evaluation"]["threshold"])
    print("ROI Size         :", tuple(config["data"]["patch_size"]))
    print("SW Batch Size    :", config["evaluation"]["sw_batch_size"])
    print("Overlap          :", config["evaluation"]["overlap"])
    print("=" * 75)

    patient_df, summary_df = evaluate_split(
        model, loader, device, config, prediction_dir
    )
    positive_df = patient_df[patient_df["gt_volume_ml"] > 0]
    empty_gt_df = patient_df[patient_df["gt_volume_ml"] == 0]
    false_positive_empty_cases = (
        empty_gt_df[
            empty_gt_df["pred_volume_ml"]
            > 0
        ]
    )
    macro_dice_all = float(patient_df["dice"].mean())
    macro_dice_positive = (
        float(positive_df["dice"].mean()) if len(positive_df) else np.nan
    )

    metadata = {
        "experiment_id": "I-S00",
        "evaluation_split": args.split,
        "checkpoint": str(checkpoint_path),
        "best_epoch": int(checkpoint["epoch"]) + 1,
        "training_best_validation_dice": float(checkpoint["best_dice"]),
        f"{args.split}_cases": len(patient_df),
        "gt_positive_cases": len(positive_df),
        "gt_empty_cases": len(empty_gt_df),
        "empty_gt_false_positive_cases": len(
            false_positive_empty_cases
        ),
        f"{args.split}_macro_dice_all": macro_dice_all,
        f"{args.split}_median_dice_all": float(patient_df["dice"].median()),
        f"{args.split}_std_dice_all": float(patient_df["dice"].std(ddof=1)),
        f"{args.split}_macro_dice_gt_positive_only": macro_dice_positive,
        "threshold": float(config["evaluation"]["threshold"]),
        "roi_size": list(config["data"]["patch_size"]),
        "sw_batch_size": int(config["evaluation"]["sw_batch_size"]),
        "overlap": float(config["evaluation"]["overlap"]),
    }

    if output_dir is not None:
        patient_df.to_csv(output_dir / f"{args.split}_patient_level_metrics.csv", index=False)
        summary_df.to_csv(output_dir / f"{args.split}_summary_metrics.csv", index=False)
        with (output_dir / f"{args.split}_evaluation_metadata.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print()
    print(f"{args.split} Cases                  : {len(patient_df)}")
    print(f"GT-positive Cases            : {len(positive_df)}")
    print(f"GT-empty Cases               : {len(empty_gt_df)}")
    print(f"Macro Dice (all cases)       : {macro_dice_all:.6f}")
    print(f"Macro Dice (GT-positive)     : {macro_dice_positive:.6f}")
    print(f"Checkpoint Best Val Dice     : {float(checkpoint['best_dice']):.6f}")


if __name__ == "__main__":
    main()
