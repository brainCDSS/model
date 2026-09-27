#!/usr/bin/env python3
"""Full-volume held-out test export for the three BrainCDSS nnU-Net baselines."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import torch
from brainon_ai.nnunet.config import set_config_environment
from brainon_ai.nnunet.runner import configure_paths
from nnunetv2.inference.export_prediction import export_prediction_from_logits
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

REPO = Path(__file__).resolve().parents[2]
DATASETS = REPO.parent / "datasets"
SPECS = {
 "H-T02-E300": dict(config=REPO/"configs/experiments/hemorrhage/tuning/H-T02-E300.yaml", data=DATASETS/"hemorrhage/v1/model_ready/nnunet", model=REPO/"runs/nnunet/Dataset001_ICHBinary/nnUNetTrainerICHBaseline__ICHPlansTuneA02B08SgdE300V1__3d_fullres", images=DATASETS/"hemorrhage/v1/model_ready/nnunet/nnUNet_raw/Dataset001_ICHBinary/imagesTs", labels=DATASETS/"hemorrhage/v1/model_ready/nnunet/nnUNet_raw/Dataset001_ICHBinary/labelsTs", checkpoint="checkpoint_best_composite.pth"),
 "I-N00": dict(config=REPO/"configs/experiments/ischemia/I-N00.yaml", data=DATASETS/"ischemic/nnunet", model=REPO/"runs/I-nnunet/I-N00", images=DATASETS/"ischemic/nnunet/nnUNet_raw/Dataset502_ISLES2022/imagesTs", labels=DATASETS/"ischemic/nnunet/nnUNet_raw/Dataset502_ISLES2022/labelsTs", checkpoint="checkpoint_best.pth"),
 "A-B04": dict(config=REPO/"configs/experiments/aneurysm_binary/A-B04.yaml", data=DATASETS/"aneurysm/v2/model_ready/nnunet", model=REPO/"runs/aneurysm_binary/nnunet/Dataset503_TopAneuMRABinary/nnUNetTrainerAneurysmBinaryBaseline__AneurysmBinaryPlansTverskyFocalV1__3d_fullres", images=DATASETS/"aneurysm/v2/model_ready/nnunet/nnUNet_raw/Dataset503_TopAneuMRABinary/imagesTs", labels=DATASETS/"aneurysm/v2/model_ready/nnunet/nnUNet_raw/Dataset503_TopAneuMRABinary/test_gt", checkpoint="checkpoint_best_full_macro.pth"),
}
def cid(path: Path) -> str: return path.name.removesuffix('.nii.gz').removesuffix('.nii')
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--model-id',choices=SPECS,nargs='+',default=['H-T02-E300','A-B04']);ap.add_argument('--limit',type=int);ap.add_argument('--overwrite',action='store_true'); a=ap.parse_args()
 if not torch.cuda.is_available(): raise RuntimeError('CUDA is required for full-volume test export')
 os.environ.setdefault('nnUNet_def_n_proc','1')
 trainer_paths=[REPO/'src/brainon_ai/nnunet/trainers',REPO/'src/brainon_ai/models/aneurysm_binary/nnunet']
 os.environ['nnUNet_extTrainer']=os.pathsep.join(map(str,trainer_paths))
 for model_id in a.model_id:
  s=SPECS[model_id]; checkpoint=s['model']/ 'fold_0'/s['checkpoint']; out=REPO/'runs/final_test_exports'/model_id/'predictions'
  if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
  labels=sorted(s['labels'].glob('*.nii.gz'))
  if not labels: raise FileNotFoundError(f'No held-out GT labels: {s["labels"]}')
  cases=[cid(x) for x in labels]; cases=cases[:a.limit] if a.limit else cases
  set_config_environment(s['config']); configure_paths(s['data'],s['model'].parent)
  predictor=nnUNetPredictor(tile_step_size=.5,use_gaussian=True,use_mirroring=True,perform_everything_on_device=True,device=torch.device('cuda'),verbose=False,verbose_preprocessing=False,allow_tqdm=True)
  predictor.initialize_from_trained_model_folder(str(s['model']),use_folds=(0,),checkpoint_name=s['checkpoint'])
  pre=predictor.configuration_manager.preprocessor_class(verbose=False); out.mkdir(parents=True,exist_ok=True)
  todo=[x for x in cases if a.overwrite or not (out/f'{x}.nii.gz').is_file()]
  print(f'===== {model_id}: test cases={len(cases)}, export={len(todo)}, skip={len(cases)-len(todo)} =====',flush=True)
  for index,case in enumerate(todo,1):
   images=sorted(s['images'].glob(f'{case}_????.nii.gz'))
   if not images: raise FileNotFoundError(f'{model_id}: images missing for {case}')
   data,_,props=pre.run_case([str(x) for x in images],None,predictor.plans_manager,predictor.configuration_manager,predictor.dataset_json)
   tensor=torch.from_numpy(data).to(dtype=torch.float32,memory_format=torch.contiguous_format)
   logits=predictor.predict_logits_from_preprocessed_data(tensor).cpu().numpy()
   export_prediction_from_logits(logits,props,predictor.configuration_manager,predictor.plans_manager,predictor.dataset_json,str(out/case),save_probabilities=False,num_threads_torch=1)
   del data,tensor,logits; torch.cuda.empty_cache(); print(f'[{model_id} {index}/{len(todo)}] {case}',flush=True)
  del predictor; torch.cuda.empty_cache()
 return 0
if __name__=='__main__': raise SystemExit(main())
