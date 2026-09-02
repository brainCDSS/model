#!/usr/bin/env python3

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from brainon_ai.nnunet.runner import configure_paths
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.export_prediction import export_prediction_from_logits
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


REPO_ROOT = Path("/home/stu03/workspace/model")
DATA_ROOT = Path(
    "/home/stu03/workspace/datasets/hemorrhage/v1/model_ready/nnunet"
)
RESULTS_ROOT = REPO_ROOT / "runs/nnunet"
DATASET = "Dataset002_BHSDOnly"
MODEL_FOLDER = (
    RESULTS_ROOT
    / DATASET
    / "nnUNetTrainerICHBaseline__ICHPlansBHSDBaselineV1__3d_fullres"
)
FOLD_FOLDER = MODEL_FOLDER / "fold_0"
VALIDATION_FOLDER = FOLD_FOLDER / "validation"
RAW_IMAGES = DATA_ROOT / "nnUNet_raw" / DATASET / "imagesTr"
GT_FOLDER = DATA_ROOT / "nnUNet_preprocessed" / DATASET / "gt_segmentations"
SPLIT_PATH = DATA_ROOT / "nnUNet_preprocessed" / DATASET / "splits_final.json"


def main() -> int:
    import json

    os.environ["ICH_NNUNET_CONFIG"] = str(
        REPO_ROOT / "configs/experiments/hemorrhage/H-N40.yaml"
    )
    configure_paths(DATA_ROOT, RESULTS_ROOT)

    validation_cases = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))[0]["val"]
    missing = [
        case
        for case in validation_cases
        if not (VALIDATION_FOLDER / f"{case}.nii.gz").is_file()
    ]
    print(f"validation cases: {len(validation_cases)}")
    print(f"missing predictions: {len(missing)}")

    if missing:
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device("cuda"),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True,
        )
        predictor.initialize_from_trained_model_folder(
            str(MODEL_FOLDER),
            use_folds=(0,),
            checkpoint_name="checkpoint_best_composite.pth",
        )
        preprocessor = predictor.configuration_manager.preprocessor_class(verbose=False)

        for index, case in enumerate(missing, start=1):
            print(f"[{index}/{len(missing)}] predicting {case}")
            data, _, properties = preprocessor.run_case(
                [str(RAW_IMAGES / f"{case}_0000.nii.gz")],
                None,
                predictor.plans_manager,
                predictor.configuration_manager,
                predictor.dataset_json,
            )
            tensor = torch.from_numpy(data).to(
                dtype=torch.float32,
                memory_format=torch.contiguous_format,
            )
            logits = (
                predictor.predict_logits_from_preprocessed_data(tensor)
                .cpu()
                .detach()
                .numpy()
            )
            export_prediction_from_logits(
                logits,
                properties,
                predictor.configuration_manager,
                predictor.plans_manager,
                predictor.dataset_json,
                str(VALIDATION_FOLDER / case),
                False,
            )
            del tensor, logits
            torch.cuda.empty_cache()

    label_manager = predictor.plans_manager.get_label_manager(predictor.dataset_json) if missing else None
    if label_manager is None:
        predictor = nnUNetPredictor(device=torch.device("cuda"))
        predictor.initialize_from_trained_model_folder(
            str(MODEL_FOLDER), use_folds=(0,), checkpoint_name="checkpoint_best_composite.pth"
        )
        label_manager = predictor.plans_manager.get_label_manager(predictor.dataset_json)

    metrics = compute_metrics_on_folder(
        str(GT_FOLDER),
        str(VALIDATION_FOLDER),
        str(VALIDATION_FOLDER / "summary.json"),
        predictor.plans_manager.image_reader_writer_class(),
        predictor.dataset_json["file_ending"],
        label_manager.foreground_regions
        if label_manager.has_regions
        else label_manager.foreground_labels,
        label_manager.ignore_label,
        num_processes=1,
        chill=True,
    )
    print("foreground mean:", metrics["foreground_mean"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
