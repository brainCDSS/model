#!/usr/bin/env python3
from pathlib import Path
import torch
from brainon_ai.nnunet.config import set_config_environment
from brainon_ai.nnunet.runner import configure_paths
from nnunetv2.run.run_training import get_trainer_from_args
ROOT=Path("/home/stu03/workspace/model")
DATA_ROOT=Path("/home/stu03/workspace/datasets/hemorrhage/v1/model_ready/nnunet")
def main():
    configure_paths(DATA_ROOT,ROOT/"runs/nnunet")
    set_config_environment(ROOT/"configs/experiments/hemorrhage/H-N130.yaml")
    trainer=get_trainer_from_args("1","3d_fullres",0,"nnUNetTrainerICHBaseline","ICHPlansCaseComponentSamplingV1",False,device=torch.device("cuda"))
    checkpoint=Path(trainer.output_folder)/"checkpoint_best_composite.pth"
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    trainer.load_checkpoint(checkpoint)
    trainer.perform_actual_validation(False)
    print("summary:",Path(trainer.output_folder)/"validation/summary.json")
    return 0
if __name__=="__main__":raise SystemExit(main())
