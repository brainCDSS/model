#!/usr/bin/env python3
"""Sweep binary thresholds for every completed hemorrhage/aneurysm nnU-Net model."""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import yaml
from scipy import ndimage


THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.05, 1.0, 0.05))
CHECKPOINT_PRIORITY = (
    "checkpoint_best_dual_criteria.pth",
    "checkpoint_best_full_macro.pth",
    "checkpoint_best_composite.pth",
    "checkpoint_best.pth",
    "checkpoint_final.pth",
)


@dataclass(frozen=True)
class ModelSpec:
    experiment: str
    disease: str
    dataset: str
    fold: int
    model_folder: str
    checkpoint: str
    config: str
    data_root: str
    results_root: str


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def discover_models(repo: Path) -> list[ModelSpec]:
    """Return completed models with an unambiguous source experiment config."""
    config_paths = [
        *repo.glob("configs/experiments/hemorrhage/**/*.yaml"),
        *repo.glob("configs/experiments/aneurysm_binary/**/*.yaml"),
    ]
    configs = []
    for path in config_paths:
        config = _read_yaml(path)
        if "experiment" in config and "plans" in config:
            configs.append((path, config))

    result_roots = (
        ("hemorrhage", repo / "runs/nnunet", repo.parent / "datasets/hemorrhage/v1/model_ready/nnunet"),
        ("aneurysm", repo / "runs/aneurysm_binary/nnunet", repo.parent / "datasets/aneurysm/v2/model_ready/nnunet"),
    )
    discovered = []
    for disease, results_root, data_root in result_roots:
        for fold_folder in results_root.glob("Dataset*/*/fold_*"):
            if not re.fullmatch(r"fold_\d+", fold_folder.name):
                continue
            model_folder = fold_folder.parent
            parts = model_folder.name.split("__")
            if len(parts) != 3:
                continue
            _, plans_identifier, configuration = parts
            dataset = model_folder.parent.name
            dataset_id = int(dataset.split("_", 1)[0].removeprefix("Dataset"))
            fold = int(fold_folder.name.removeprefix("fold_"))
            complete = (
                (fold_folder / "checkpoint_final.pth").is_file()
                or (fold_folder / "final_selection.json").is_file()
            )
            checkpoint = next(
                (
                    name
                    for name in CHECKPOINT_PRIORITY
                    if (fold_folder / name).is_file()
                ),
                None,
            )
            if not complete or checkpoint is None:
                continue
            matches = [
                (path, config)
                for path, config in configs
                if int(config["experiment"]["dataset_id"]) == dataset_id
                and str(config["experiment"].get("configuration")) == configuration
                and int(config["experiment"].get("fold", 0)) == fold
                and str(config["plans"]["output_identifier"]) == plans_identifier
            ]
            if len(matches) != 1:
                print(
                    f"SKIP config match={len(matches)}: {model_folder}",
                    file=sys.stderr,
                )
                continue
            config_path, _ = matches[0]
            discovered.append(
                ModelSpec(
                    experiment=config_path.stem,
                    disease=disease,
                    dataset=dataset,
                    fold=fold,
                    model_folder=str(model_folder),
                    checkpoint=checkpoint,
                    config=str(config_path.resolve()),
                    data_root=str(data_root.resolve()),
                    results_root=str(results_root.resolve()),
                )
            )
    return sorted(discovered, key=lambda item: (item.disease, item.experiment, item.fold))


def _empty_accumulator() -> dict[str, float | int]:
    return {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "positive_cases": 0,
        "macro_dice_sum": 0.0,
        "macro_recall_sum": 0.0,
        "macro_precision_sum": 0.0,
        "negative_cases": 0,
        "negative_cases_with_fp": 0,
        "negative_fp_components": 0,
        "total_lesions": 0,
        "detected_lesions": 0,
        "lesion_dice_sum": 0.0,
        "detected_lesion_dice_sum": 0.0,
    }


def _update_metrics(
    accumulator: dict,
    gt: np.ndarray,
    pred: np.ndarray,
    structure: np.ndarray,
) -> None:
    tp = int(np.count_nonzero(gt & pred))
    fp = int(np.count_nonzero(~gt & pred))
    fn = int(np.count_nonzero(gt & ~pred))
    accumulator["tp"] += tp
    accumulator["fp"] += fp
    accumulator["fn"] += fn

    if gt.any():
        accumulator["positive_cases"] += 1
        accumulator["macro_dice_sum"] += 2.0 * tp / (2 * tp + fp + fn)
        accumulator["macro_recall_sum"] += tp / (tp + fn)
        accumulator["macro_precision_sum"] += tp / (tp + fp) if tp + fp else 0.0
    else:
        accumulator["negative_cases"] += 1
        pred_labels, pred_count = ndimage.label(pred, structure=structure)
        accumulator["negative_fp_components"] += int(pred_count)
        accumulator["negative_cases_with_fp"] += int(pred_count > 0)
        del pred_labels

    gt_labels, gt_count = ndimage.label(gt, structure=structure)
    if gt_count == 0:
        return
    pred_labels, _ = ndimage.label(pred, structure=structure)
    for lesion_id, lesion_slice in enumerate(ndimage.find_objects(gt_labels), start=1):
        if lesion_slice is None:
            continue
        gt_component = gt_labels[lesion_slice] == lesion_id
        overlapping = np.unique(pred_labels[lesion_slice][gt_component])
        overlapping = overlapping[overlapping > 0]
        detected = len(overlapping) > 0
        lesion_dice = 0.0
        if detected:
            pred_component = np.isin(pred_labels, overlapping)
            intersection = int(
                np.count_nonzero(pred_component & (gt_labels == lesion_id))
            )
            denominator = int(pred_component.sum()) + int(gt_component.sum())
            lesion_dice = 2.0 * intersection / denominator if denominator else 0.0
        accumulator["total_lesions"] += 1
        accumulator["detected_lesions"] += int(detected)
        accumulator["lesion_dice_sum"] += lesion_dice
        if detected:
            accumulator["detected_lesion_dice_sum"] += lesion_dice
    del gt_labels, pred_labels


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _finalize(spec: ModelSpec, threshold: float, values: dict) -> dict:
    tp, fp, fn = int(values["tp"]), int(values["fp"]), int(values["fn"])
    positive = int(values["positive_cases"])
    negative = int(values["negative_cases"])
    lesions = int(values["total_lesions"])
    detected = int(values["detected_lesions"])
    return {
        "experiment": spec.experiment,
        "disease": spec.disease,
        "dataset": spec.dataset,
        "fold": spec.fold,
        "checkpoint": spec.checkpoint,
        "threshold": threshold,
        "positive_cases": positive,
        "negative_cases": negative,
        "positive_macro_dice": _safe_ratio(values["macro_dice_sum"], positive),
        "positive_macro_recall": _safe_ratio(values["macro_recall_sum"], positive),
        "positive_macro_precision": _safe_ratio(values["macro_precision_sum"], positive),
        "voxel_micro_dice": _safe_ratio(2 * tp, 2 * tp + fp + fn),
        "voxel_micro_recall": _safe_ratio(tp, tp + fn),
        "voxel_micro_precision": _safe_ratio(tp, tp + fp),
        "tp_voxels": tp,
        "fp_voxels": fp,
        "fn_voxels": fn,
        "total_lesions": lesions,
        "detected_lesions": detected,
        "lesion_detection_rate": _safe_ratio(detected, lesions),
        "mean_lesion_dice_including_missed": _safe_ratio(values["lesion_dice_sum"], lesions),
        "mean_lesion_dice_detected_only": _safe_ratio(
            values["detected_lesion_dice_sum"], detected
        ),
        "negative_case_any_fp_rate": _safe_ratio(
            values["negative_cases_with_fp"], negative
        ),
        "mean_fp_components_per_negative_case": _safe_ratio(
            values["negative_fp_components"], negative
        ),
    }


def _raw_images_for_case(raw_images: Path, case: str, channels: int) -> list[str]:
    paths = [raw_images / f"{case}_{index:04d}.nii.gz" for index in range(channels)]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("입력 영상 없음: " + ", ".join(map(str, missing)))
    return [str(path) for path in paths]


def evaluate_model(
    spec: ModelSpec,
    output_root: Path,
    *,
    device: str,
    perform_everything_on_device: bool,
    use_mirroring: bool,
) -> Path:
    """Run one model and atomically write its 19-row threshold CSV."""
    from brainon_ai.nnunet.config import set_config_environment
    from brainon_ai.nnunet.runner import configure_paths
    from nnunetv2.inference.export_prediction import export_prediction_from_logits
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    output = output_root / "per_model" / f"{spec.experiment}_fold{spec.fold}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        existing = pd.read_csv(output)
        if len(existing) == len(THRESHOLDS) and set(existing["threshold"].round(2)) == set(THRESHOLDS):
            print(f"SKIP completed: {spec.experiment} fold {spec.fold}", flush=True)
            return output

    data_root = Path(spec.data_root)
    results_root = Path(spec.results_root)
    dataset_dir = data_root / "nnUNet_preprocessed" / spec.dataset
    split_path = dataset_dir / "splits_final.json"
    dataset_json = json.loads((dataset_dir / "dataset.json").read_text(encoding="utf-8"))
    channels = len(dataset_json["channel_names"])
    cases = json.loads(split_path.read_text(encoding="utf-8"))[spec.fold]["val"]
    raw_images = data_root / "nnUNet_raw" / spec.dataset / "imagesTr"
    gt_folder = dataset_dir / "gt_segmentations"

    set_config_environment(spec.config)
    configure_paths(data_root, results_root)
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=use_mirroring,
        perform_everything_on_device=perform_everything_on_device,
        device=torch.device(device),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        spec.model_folder,
        use_folds=(spec.fold,),
        checkpoint_name=spec.checkpoint,
    )
    preprocessor = predictor.configuration_manager.preprocessor_class(verbose=False)
    accumulators = {threshold: _empty_accumulator() for threshold in THRESHOLDS}
    structure = ndimage.generate_binary_structure(rank=3, connectivity=3)

    with tempfile.TemporaryDirectory(prefix=f"threshold_{spec.experiment}_") as temporary:
        temporary_path = Path(temporary)
        for index, case in enumerate(cases, start=1):
            print(
                f"[{spec.experiment} fold {spec.fold} {index}/{len(cases)}] {case}",
                flush=True,
            )
            inputs = _raw_images_for_case(raw_images, case, channels)
            data, _, properties = preprocessor.run_case(
                inputs,
                None,
                predictor.plans_manager,
                predictor.configuration_manager,
                predictor.dataset_json,
            )
            tensor = torch.from_numpy(data).to(
                dtype=torch.float32, memory_format=torch.contiguous_format
            )
            logits = (
                predictor.predict_logits_from_preprocessed_data(tensor)
                .cpu()
                .detach()
                .numpy()
            )
            output_base = temporary_path / case
            export_prediction_from_logits(
                logits,
                properties,
                predictor.configuration_manager,
                predictor.plans_manager,
                predictor.dataset_json,
                str(output_base),
                save_probabilities=True,
                num_threads_torch=1,
            )
            with np.load(output_base.with_suffix(".npz")) as probability_file:
                foreground_probability = probability_file["probabilities"][1]
            # nibabel exposes NIfTI arrays as (x, y, z), while nnU-Net exports
            # probabilities in array order (z, y, x).
            gt_xyz = np.asanyarray(
                nib.load(gt_folder / f"{case}.nii.gz").dataobj
            ) > 0
            gt = np.transpose(gt_xyz, (2, 1, 0))
            if foreground_probability.shape != gt.shape:
                raise ValueError(
                    f"shape 불일치 {case}: probability={foreground_probability.shape}, "
                    f"GT xyz={gt_xyz.shape}, GT zyx={gt.shape}"
                )
            for threshold in THRESHOLDS:
                _update_metrics(
                    accumulators[threshold],
                    gt,
                    foreground_probability >= threshold,
                    structure,
                )
            for generated in temporary_path.glob(f"{case}*"):
                generated.unlink()
            del data, tensor, logits, foreground_probability, gt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    rows = [_finalize(spec, threshold, accumulators[threshold]) for threshold in THRESHOLDS]
    temporary_csv = output.with_suffix(".csv.partial")
    pd.DataFrame(rows).to_csv(temporary_csv, index=False)
    temporary_csv.replace(output)
    del predictor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def aggregate_results(output_root: Path) -> tuple[Path, Path]:
    files = sorted((output_root / "per_model").glob("*.csv"))
    if not files:
        raise RuntimeError("완료된 threshold 결과 CSV가 없습니다")
    combined = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    combined_path = output_root / "all_models_threshold_metrics.csv"
    combined.to_csv(combined_path, index=False)

    selected_rows = []
    for (_, _, fold), group in combined.groupby(["disease", "experiment", "fold"]):
        best_macro = float(group["positive_macro_dice"].max())
        eligible = group[group["positive_macro_dice"] >= best_macro - 0.01]
        selected = eligible.sort_values(
            [
                "lesion_detection_rate",
                "positive_macro_dice",
                "negative_case_any_fp_rate",
                "threshold",
            ],
            ascending=[False, False, True, False],
        ).iloc[0].copy()
        selected["threshold_selection_rule"] = (
            "macro Dice within 0.01 of best; maximize lesion detection; "
            "then minimize negative-case FP; then higher threshold"
        )
        selected_rows.append(selected)
    best_path = output_root / "selected_thresholds.csv"
    pd.DataFrame(selected_rows).sort_values(
        ["disease", "experiment", "fold"]
    ).to_csv(best_path, index=False)
    return combined_path, best_path


def configure_external_trainers(repo: Path) -> None:
    source = repo / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    paths = (
        repo / "src/brainon_ai/nnunet/trainers",
        repo / "src/brainon_ai/models/aneurysm_binary/nnunet",
    )
    os.environ["nnUNet_extTrainer"] = os.pathsep.join(map(str, paths))
    os.environ.setdefault("nnUNet_def_n_proc", "1")
    os.environ.setdefault("nnUNet_compile", "false")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--everything-on-device", action="store_true")
    parser.add_argument(
        "--no-mirroring",
        action="store_true",
        help="Disable test-time mirroring for a faster first-pass threshold sweep.",
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    configure_external_trainers(repo)
    specs = discover_models(repo)
    if args.only:
        requested = set(args.only)
        specs = [spec for spec in specs if spec.experiment in requested]
    print(pd.DataFrame(asdict(spec) for spec in specs).to_string(index=False))
    print(f"models={len(specs)}, thresholds={len(THRESHOLDS)}")
    if args.list:
        return 0
    output_root = (args.output_root or repo / "runs/threshold_sweeps/all_completed_models").resolve()
    for index, spec in enumerate(specs, start=1):
        print(f"===== [{index}/{len(specs)}] {spec.experiment} fold {spec.fold} =====", flush=True)
        evaluate_model(
            spec,
            output_root,
            device=args.device,
            perform_everything_on_device=args.everything_on_device,
            use_mirroring=not args.no_mirroring,
        )
        aggregate_results(output_root)
    combined, selected = aggregate_results(output_root)
    print(f"all metrics: {combined}")
    print(f"selected thresholds: {selected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
