#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os
from pathlib import Path
import nibabel as nib
import numpy as np
import torch

def segresnet(model,repo,overwrite):
 from monai.data import DataLoader,Dataset
 from brainon_ai.models.ischemia.segresnet_hy.checkpointing import load_checkpoint,load_model_state
 from brainon_ai.models.ischemia.segresnet_hy.data import build_records,test_transform
 from brainon_ai.models.ischemia.segresnet_hy.inference import infer_probability
 from brainon_ai.models.ischemia.segresnet_hy.network import build_model
 checkpoint=load_checkpoint(Path(model['checkpoint']),torch.device('cuda')); cfg=checkpoint['config']; net=build_model(cfg).cuda(); load_model_state(net,checkpoint); net.eval(); root=Path(model['data_root']); loader=DataLoader(Dataset(build_records(root,'val',include_label_path=True),transform=test_transform()),batch_size=1,shuffle=False,num_workers=int(cfg['data']['num_workers']))
 out=Path(model['tta_prediction_dirs'][0]); out.mkdir(parents=True,exist_ok=True)
 for i,b in enumerate(loader,1):
  if model.get('_limit') and i>model['_limit']: break
  cid=b['case_id'][0]; dest=out/f'{cid}.npz'
  if dest.is_file() and not overwrite: print(f'[{i}/{len(loader)}] skip {cid}'); continue
  affine=nib.load(b['label_path'][0]).affine; codes=nib.aff2axcodes(affine); axes=[j for j,c in enumerate(codes) if c in ('L','R')]
  if len(axes)!=1: raise RuntimeError(f'{cid}: cannot identify one LR axis from {codes}')
  axis=axes[0]; image=b['image'].cuda(); flipped=torch.flip(image,dims=[2+axis]); prob=infer_probability(net,flipped,tuple(cfg['data']['patch_size']),int(cfg['evaluation']['sw_batch_size']),float(cfg['evaluation']['overlap']),False); prob=torch.flip(prob,dims=[2+axis])[0,0].cpu().numpy().astype(np.float32); np.savez_compressed(dest,probabilities=np.stack([1-prob,prob])); print(f'[{i}/{len(loader)}] {cid} LR-axis={axis}',flush=True)

def nnunet(model,repo,overwrite):
 from brainon_ai.nnunet.config import set_config_environment
 from brainon_ai.nnunet.runner import configure_paths
 from nnunetv2.inference.export_prediction import export_prediction_from_logits
 from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
 set_config_environment(model['config']); configure_paths(Path(model['data_root']),Path(model['results_root'])); os.environ.setdefault('nnUNet_def_n_proc','1'); os.environ['nnUNet_extTrainer']=os.pathsep.join([str(repo/'src/brainon_ai/nnunet/trainers'),str(repo/'src/brainon_ai/models/aneurysm_binary/nnunet')])
 predictor=nnUNetPredictor(tile_step_size=.5,use_gaussian=True,use_mirroring=True,perform_everything_on_device=True,device=torch.device('cuda'),verbose=False,verbose_preprocessing=False,allow_tqdm=True); predictor.initialize_from_trained_model_folder(model['model_folder'],use_folds=(0,),checkpoint_name=model['checkpoint_name'])
 axes=tuple(predictor.allowed_mirroring_axes or ());
 if not axes: raise RuntimeError('trained nnU-Net has no allowed mirroring axis')
 axis=int(axes[0]); split=json.loads(Path(model['split_file']).read_text())[0]['val']; split=split[:model['_limit']] if model.get('_limit') else split; raw=Path(model['raw_images']); original=Path(model['prediction_dir']); tta=Path(model['tta_prediction_dirs'][0]); original.mkdir(parents=True,exist_ok=True); tta.mkdir(parents=True,exist_ok=True); prep=predictor.configuration_manager.preprocessor_class(verbose=False)
 for i,cid in enumerate(split,1):
  images=sorted(raw.glob(f'{cid}_????.nii.gz'))
  if not images: raise FileNotFoundError(f'images for {cid}')
  tdest=tta/f'{cid}.npz'; need_original=overwrite or not (original/f'{cid}.npz').is_file(); need_tta=overwrite or not tdest.is_file()
  if not need_original and not need_tta: print(f'[{i}/{len(split)}] skip {cid}'); continue
  data,_,props=prep.run_case([str(x) for x in images],None,predictor.plans_manager,predictor.configuration_manager,predictor.dataset_json); tensor=torch.from_numpy(data).float().contiguous()
  if need_original:
   predictor.use_mirroring=True
   logits=predictor.predict_logits_from_preprocessed_data(tensor).numpy(); export_prediction_from_logits(logits,props,predictor.configuration_manager,predictor.plans_manager,predictor.dataset_json,str(original/cid),save_probabilities=True,num_threads_torch=1); del logits
  if need_tta:
   predictor.use_mirroring=False
   flipped=torch.flip(tensor,dims=[axis+1]); logits=predictor.predict_logits_from_preprocessed_data(flipped); logits=torch.flip(logits,dims=[axis+1]).numpy(); export_prediction_from_logits(logits,props,predictor.configuration_manager,predictor.plans_manager,predictor.dataset_json,str(tta/cid),save_probabilities=True,num_threads_torch=1); del logits,flipped
  del tensor,data; torch.cuda.empty_cache(); print(f'[{i}/{len(split)}] {cid} preprocessed-axis={axis}',flush=True)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--model-id',required=True); ap.add_argument('--registry',type=Path,default=Path('configs/calibration/models.yaml')); ap.add_argument('--overwrite',action='store_true'); ap.add_argument('--limit',type=int); a=ap.parse_args(); repo=Path(__file__).resolve().parents[2]
 from brainon_ai.calibration.registry import resolve_model
 m=resolve_model(a.model_id,a.registry,repo); m['_limit']=a.limit
 if not torch.cuda.is_available(): raise RuntimeError('CUDA required for TTA inference')
 (segresnet if m.get('backend')=='segresnet' else nnunet)(m,repo,a.overwrite)
if __name__=='__main__': main()
