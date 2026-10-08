#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import torch

from brainon_ai.nnunet.config import set_config_environment
from brainon_ai.nnunet.runner import configure_paths
from nnunetv2.run.run_training import get_trainer_from_args

REPOSITORY = Path("/home/stu03/workspace/model")
DATA_ROOT = Path("/home/stu03/workspace/datasets/hemorrhage/v1/model_ready/nnunet")
EXPERIMENTS = (
    ("H-N60", 1, "ICHPlansBaselineSmallLesionV1"),
    ("H-N70", 3, "ICHPlansTwoWindowSmallLesionV1"),
)


def main() -> int:
    configure_paths(DATA_ROOT, REPOSITORY / "runs/nnunet")
    for name, dataset_id, plans in EXPERIMENTS:
        print(f"===== {name} full-volume validation =====", flush=True)
        set_config_environment(REPOSITORY / f"configs/experiments/hemorrhage/{name}.yaml")
        trainer = get_trainer_from_args(
            str(dataset_id), "3d_fullres", 0, "nnUNetTrainerICHBaseline", plans,
            False, device=torch.device("cuda")
        )
        trainer.load_checkpoint(Path(trainer.output_folder) / "checkpoint_best_composite.pth")
        trainer.perform_actual_validation(False)
        del trainer
        torch.cuda.empty_cache()
        print(f"===== {name} completed =====", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
