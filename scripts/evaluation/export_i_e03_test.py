#!/usr/bin/env python3
"""Fixed I-E03 test export: N120 .6 + S101 .2 + S50 .2, threshold .20."""
from __future__ import annotations
import argparse, os
from pathlib import Path
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import torch
from monai.data import DataLoader, Dataset
from brainon_ai.nnunet.config import set_config_environment
from brainon_ai.nnunet.runner import configure_paths
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.export_prediction import export_prediction_from_logits
from itertools import permutations
from brainon_ai.models.ischemia.segresnet_hy.checkpointing import load_checkpoint, load_model_state
from brainon_ai.models.ischemia.segresnet_hy.data import build_records, test_transform
from brainon_ai.models.ischemia.segresnet_hy.inference import infer_probability
from brainon_ai.models.ischemia.segresnet_hy.network import build_model
REPO=Path(__file__).resolve().parents[2]; DATA=REPO.parent/'datasets/ischemic'; OUT=REPO/'runs/final_test_exports/I-E03/predictions'; SCORE_OUT=REPO/'runs/score_external_validation/I-E03'
def n120_probs(limit):
 config=REPO/'configs/experiments/ischemia/I-N120.yaml'; model=REPO/'runs/I-nnunet/Dataset502_ISLES2022/nnUNetTrainerIschemiaMultimodal__IschemiaPlansDualEncoderGatedV1__3d_fullres'; raw=DATA/'nnunet/nnUNet_raw/Dataset502_ISLES2022/imagesTs'; labels=DATA/'nnunet/nnUNet_raw/Dataset502_ISLES2022/labelsTs'
 set_config_environment(config); configure_paths(DATA/'nnunet',model.parent.parent.parent.parent); os.environ['nnUNet_extTrainer']=str(REPO/'src/brainon_ai/nnunet/trainers')
 pr=nnUNetPredictor(tile_step_size=.5,use_gaussian=True,use_mirroring=True,perform_everything_on_device=True,device=torch.device('cuda'),verbose=False,verbose_preprocessing=False,allow_tqdm=False); pr.initialize_from_trained_model_folder(str(model),use_folds=(0,),checkpoint_name='checkpoint_best_full_macro.pth'); pre=pr.configuration_manager.preprocessor_class(verbose=False)
 ids=[x.name[:-7] for x in sorted(labels.glob('*.nii.gz'))]; ids=ids[:limit] if limit else ids
 out={}
 for i,cid in enumerate(ids,1):
  images=sorted(raw.glob(f'{cid}_????.nii.gz')); data,_,props=pre.run_case([str(x) for x in images],None,pr.plans_manager,pr.configuration_manager,pr.dataset_json)
  logits=pr.predict_logits_from_preprocessed_data(torch.from_numpy(data).float().contiguous()).cpu().numpy()
  component=OUT/'components'/'N120'; component.mkdir(parents=True,exist_ok=True)
  export_prediction_from_logits(logits,props,pr.configuration_manager,pr.plans_manager,pr.dataset_json,str(component/cid),save_probabilities=True,num_threads_torch=1)
  with np.load(component/f'{cid}.npz') as z: probability=np.asarray(z['probabilities'][1],dtype=np.float32)
  saved=np.asanyarray(nib.load(str(component/f'{cid}.nii.gz')).dataobj)>0
  candidates=[probability.transpose(axis) for axis in permutations(range(3)) if probability.transpose(axis).shape==saved.shape]
  if len(candidates)==1: out[cid]=candidates[0]
  else:
   agreement=[np.mean((candidate>=.5)==saved) for candidate in candidates]; out[cid]=candidates[int(np.argmax(agreement))]
   if max(agreement)<.99999: raise RuntimeError(f'{cid}: cannot align N120 probability to exported NIfTI')
  del logits; print(f'[N120 {i}/{len(ids)}] {cid}',flush=True)
 return out
def seg_probs(run,limit):
 ck=load_checkpoint(REPO/f'runs/I-segresnet/{run}/checkpoints/checkpoint_best_dual_criteria.pth',torch.device('cuda')); cfg=ck['config']; net=build_model(cfg).cuda();load_model_state(net,ck);net.eval(); records=build_records(DATA/'SegResNet','test',include_label_path=True); records=records[:limit] if limit else records; out={}
 for i,b in enumerate(DataLoader(Dataset(records,transform=test_transform()),batch_size=1,shuffle=False,num_workers=0),1):
  cid=b['case_id'][0]; out[cid]=infer_probability(net,b['image'].cuda(),tuple(cfg['data']['patch_size']),int(cfg['evaluation']['sw_batch_size']),float(cfg['evaluation']['overlap']),False)[0,0].cpu().numpy(); print(f'[{run} {i}/{len(records)}] {cid}',flush=True)
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int);ap.add_argument('--overwrite',action='store_true');a=ap.parse_args()
 if not torch.cuda.is_available():raise RuntimeError('CUDA required')
 n=n120_probs(a.limit);s101=seg_probs('I-S101',a.limit);s50=seg_probs('I-S50',a.limit);OUT.mkdir(parents=True,exist_ok=True); labels=DATA/'SegResNet/test/labels'
 for cid in n:
  if cid not in s101 or cid not in s50:raise KeyError(f'{cid}: modality test cohorts differ')
  dest=OUT/f'{cid}.nii.gz'; score_prob=SCORE_OUT/'probabilities'/f'{cid}.npz'; score_mask=SCORE_OUT/'predictions'/f'{cid}.nii.gz'
  if dest.exists() and score_prob.exists() and score_mask.exists() and not a.overwrite:continue
  ref=nib.load(str(labels/f'{cid}_mask.nii.gz'))
  nn_ref=nib.load(str(OUT/'components'/'N120'/f'{cid}.nii.gz'))
  nn_prob=n[cid]
  if nn_prob.shape != ref.shape or not np.allclose(nn_ref.affine,ref.affine,atol=1e-4):
   nn_prob=np.asanyarray(resample_from_to(nib.Nifti1Image(nn_prob,nn_ref.affine),ref,order=1).dataobj)
  if s101[cid].shape != ref.shape or s50[cid].shape != ref.shape: raise ValueError(f'{cid}: SegResNet probability grid differs from its label grid')
  prob=np.clip(.6*nn_prob+.2*s101[cid]+.2*s50[cid],0,1)
  mask=nib.Nifti1Image((prob>=.2).astype(np.uint8),ref.affine,ref.header)
  dest.parent.mkdir(parents=True,exist_ok=True); SCORE_OUT.joinpath('probabilities').mkdir(parents=True,exist_ok=True); SCORE_OUT.joinpath('predictions').mkdir(parents=True,exist_ok=True)
  nib.save(mask,str(dest)); nib.save(mask,str(score_mask)); temporary=score_prob.with_name(score_prob.stem+'.writing.npz'); np.savez_compressed(temporary,foreground=np.asarray(prob,np.float32)); temporary.replace(score_prob)
 return 0
if __name__=='__main__':raise SystemExit(main())
