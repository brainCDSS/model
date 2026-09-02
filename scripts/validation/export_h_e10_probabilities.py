#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from brainon_ai.nnunet.config import set_config_environment
from brainon_ai.nnunet.runner import configure_paths
from nnunetv2.inference.export_prediction import export_prediction_from_logits
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

EXPERIMENTS = {
    "H-N00": "ICHPlansBaselineV1",
    "H-N30": "ICHPlansDefaultAttentionV1",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="H-E10용 H-N00/H-N30 validation probability 생성")
    parser.add_argument("--only", choices=EXPERIMENTS, nargs="+", default=list(EXPERIMENTS))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    data_root = repo.parent / "datasets/hemorrhage/v1/model_ready/nnunet"
    results_root = repo / "runs/nnunet"
    output_root = repo / "runs/ensembles/H-E10/probabilities"
    dataset = "Dataset001_ICHBinary"
    split_path = data_root / "nnUNet_preprocessed" / dataset / "splits_final.json"
    raw_images = data_root / "nnUNet_raw" / dataset / "imagesTr"
    cases = json.loads(split_path.read_text(encoding="utf-8"))[0]["val"]
    configure_paths(data_root, results_root)
    os.environ.setdefault("nnUNet_def_n_proc", "1")

    for experiment in args.only:
        print(f"===== {experiment}: probability export =====", flush=True)
        config_path = repo / f"configs/experiments/hemorrhage/{experiment}.yaml"
        set_config_environment(config_path)
        model_folder = results_root / dataset / (
            f"nnUNetTrainerICHBaseline__{EXPERIMENTS[experiment]}__3d_fullres"
        )
        checkpoint = model_folder / "fold_0/checkpoint_best_composite.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        destination = output_root / experiment
        destination.mkdir(parents=True, exist_ok=True)
        missing = [case for case in cases if args.overwrite or not (destination / f"{case}.npz").is_file()]
        print(f"cases={len(cases)}, export={len(missing)}, skip={len(cases)-len(missing)}", flush=True)
        if not missing:
            continue
        predictor = nnUNetPredictor(
            tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
            perform_everything_on_device=True, device=torch.device("cuda"),
            verbose=False, verbose_preprocessing=False, allow_tqdm=True,
        )
        predictor.initialize_from_trained_model_folder(
            str(model_folder), use_folds=(0,), checkpoint_name="checkpoint_best_composite.pth"
        )
        preprocessor = predictor.configuration_manager.preprocessor_class(verbose=False)
        for index, case in enumerate(missing, 1):
            print(f"[{experiment} {index}/{len(missing)}] {case}", flush=True)
            image = raw_images / f"{case}_0000.nii.gz"
            if not image.is_file():
                raise FileNotFoundError(image)
            data, _, properties = preprocessor.run_case(
                [str(image)], None, predictor.plans_manager,
                predictor.configuration_manager, predictor.dataset_json,
            )
            tensor = torch.from_numpy(data).to(dtype=torch.float32, memory_format=torch.contiguous_format)
            logits = predictor.predict_logits_from_preprocessed_data(tensor).cpu().detach().numpy()
            export_prediction_from_logits(
                logits, properties, predictor.configuration_manager, predictor.plans_manager,
                predictor.dataset_json, str(destination / case), save_probabilities=True,
                num_threads_torch=1,
            )
            del tensor, logits, data
            torch.cuda.empty_cache()
        del predictor
        torch.cuda.empty_cache()
        print(f"===== {experiment}: completed =====", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
