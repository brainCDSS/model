#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,shutil
from pathlib import Path
from itertools import permutations
import nibabel as nib
import numpy as np
import torch

def segresnet(model,repo,overwrite):
 from monai.data import DataLoader,Dataset
 from brainon_ai.models.ischemia.segresnet_hy.checkpointing import load_checkpoint,load_model_state
 from brainon_ai.models.ischemia.segresnet_hy.data import build_records,test_transform
 from brainon_ai.models.ischemia.segresnet_hy.inference import infer_probability
 from brainon_ai.models.ischemia.segresnet_hy.network import build_model
 checkpoint=load_checkpoint(Path(model['checkpoint']),torch.device('cuda')); cfg=checkpoint['config']; net=build_model(cfg).cuda(); load_model_state(net,checkpoint); net.eval(); root=Path(model['data_root']); loader=DataLoader(Dataset(build_records(root,model.get('data_split','val'),include_label_path=True),transform=test_transform()),batch_size=1,shuffle=False,num_workers=int(cfg['data']['num_workers']))
 out=Path(model['tta_prediction_dirs'][0]); out.mkdir(parents=True,exist_ok=True)
 for i,b in enumerate(loader,1):
  if model.get('_limit') and i>model['_limit']: break
  cid=b['case_id'][0]; dest=out/f'{cid}.npz'
  if dest.is_file() and not overwrite: print(f'[{i}/{len(loader)}] skip {cid}'); continue
  affine=nib.load(b['label_path'][0]).affine; codes=nib.aff2axcodes(affine); axes=[j for j,c in enumerate(codes) if c in ('L','R')]
  if len(axes)!=1: raise RuntimeError(f'{cid}: cannot identify one LR axis from {codes}')
  axis=axes[0]; image=b['image'].cuda(); flipped=torch.flip(image,dims=[2+axis]); prob=infer_probability(net,flipped,tuple(cfg['data']['patch_size']),int(cfg['evaluation']['sw_batch_size']),float(cfg['evaluation']['overlap']),False); prob=torch.flip(prob,dims=[2+axis])[0,0].cpu().numpy().astype(np.float32); np.savez_compressed(dest,probabilities=np.stack([1-prob,prob])); print(f'[{i}/{len(loader)}] {cid} LR-axis={axis}',flush=True)

def canonicalize_probability(npz_path: Path, mask_path: Path, threshold: float) -> None:
 """Put foreground probability into the affine/orientation of exported NIfTI mask."""
 with np.load(npz_path) as bundle:
  key='foreground' if 'foreground' in bundle else 'probabilities' if 'probabilities' in bundle else 'softmax'
  array=np.asarray(bundle[key])
 foreground=array[-1] if array.ndim==4 else array
 mask=np.asanyarray(nib.load(str(mask_path)).dataobj)>0
 if foreground.shape==mask.shape:
  aligned=foreground
 else:
  candidates=[foreground.transpose(order) for order in permutations(range(3)) if foreground.transpose(order).shape==mask.shape]
  if not candidates: raise ValueError(f'{npz_path}: probability {foreground.shape} cannot match mask {mask.shape}')
  agreement=[np.mean((candidate>=threshold)==mask) for candidate in candidates]
  aligned=candidates[int(np.argmax(agreement))]
  if max(agreement)<.99: raise ValueError(f'{npz_path}: ambiguous probability orientation; best mask agreement={max(agreement):.6f}')
 aligned=np.asarray(aligned,dtype=np.float32)
 # Calibration consumes foreground only. Atomic replacement avoids corrupting a
 # large NPZ if this CPU-only normalization is interrupted.
 temporary=npz_path.with_name(npz_path.stem + '.normalizing.npz')
 np.savez_compressed(temporary, foreground=aligned)
 temporary.replace(npz_path)

def normalize_existing_exports(model: dict) -> None:
 """CPU-only repair for already exported nnU-Net NPZ orientation and mask layout."""
 original=Path(model['prediction_dir']); tta_dirs=[Path(x) for x in model.get('tta_prediction_dirs', [])]
 predictions=original.parent/'predictions'; predictions.mkdir(parents=True,exist_ok=True)
 count=0
 for prob in sorted(original.glob('*.npz')):
  cid=prob.name.removesuffix('.npz'); mask=original/f'{cid}.nii.gz'
  if not mask.is_file(): raise FileNotFoundError(mask)
  canonicalize_probability(prob,mask,float(model['threshold']))
  shutil.copy2(mask,predictions/f'{cid}.nii.gz')
  for directory in tta_dirs:
   tta=directory/f'{cid}.npz'; tta_mask=directory/f'{cid}.nii.gz'
   if not tta.is_file() or not tta_mask.is_file(): raise FileNotFoundError(f'Missing TTA export for {cid}: {tta}')
   canonicalize_probability(tta,tta_mask,float(model['threshold']))
  count+=1
 print(f'Normalized {count} original-space probability/TTA export(s).')

def nnunet(model,repo,overwrite):
 from brainon_ai.nnunet.config import set_config_environment
 from brainon_ai.nnunet.runner import configure_paths
 from nnunetv2.inference.export_prediction import export_prediction_from_logits,convert_predicted_logits_to_segmentation_with_correct_shape
 from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
 set_config_environment(model['config']); configure_paths(Path(model['data_root']),Path(model['results_root'])); os.environ.setdefault('nnUNet_def_n_proc','1'); os.environ['nnUNet_extTrainer']=os.pathsep.join([str(repo/'src/brainon_ai/nnunet/trainers'),str(repo/'src/brainon_ai/models/aneurysm_binary/nnunet')])
 predictor=nnUNetPredictor(tile_step_size=.5,use_gaussian=True,use_mirroring=True,perform_everything_on_device=True,device=torch.device('cuda'),verbose=False,verbose_preprocessing=False,allow_tqdm=True); predictor.initialize_from_trained_model_folder(model['model_folder'],use_folds=(0,),checkpoint_name=model['checkpoint_name'])
 axes=tuple(predictor.allowed_mirroring_axes or ());
 if not axes: raise RuntimeError('trained nnU-Net has no allowed mirroring axis')
 axis=int(axes[0]); split=([x.name.removesuffix('.nii.gz').removesuffix('_mask') for x in sorted(Path(model['label_dir']).glob('*.nii.gz'))] if model.get('case_ids_from_labels') else json.loads(Path(model['split_file']).read_text())[0]['val']); split=split[:model['_limit']] if model.get('_limit') else split; raw=Path(model['raw_images']); original=Path(model['prediction_dir']); tta=Path(model['tta_prediction_dirs'][0]); original.mkdir(parents=True,exist_ok=True); tta.mkdir(parents=True,exist_ok=True); prep=predictor.configuration_manager.preprocessor_class(verbose=False)
 def export_deferred(logits,props,destination):
  label_manager=predictor.plans_manager.get_label_manager(predictor.dataset_json)
  segmentation,probabilities=convert_predicted_logits_to_segmentation_with_correct_shape(logits,predictor.plans_manager,predictor.configuration_manager,label_manager,props,return_probabilities=True,num_threads_torch=1)
  destination=Path(destination); temporary=destination.with_name(destination.name+'.writing.npz'); np.savez(temporary,foreground=np.asarray(probabilities[-1],np.float32)); temporary.replace(destination.with_suffix('.npz'))
  predictor.plans_manager.image_reader_writer_class().write_seg(segmentation,str(destination)+predictor.dataset_json['file_ending'],props)
 for i,cid in enumerate(split,1):
  images=sorted(raw.glob(f'{cid}_????.nii.gz'))
  if not images: raise FileNotFoundError(f'images for {cid}')
  tdest=tta/f'{cid}.npz'; need_original=overwrite or not (original/f'{cid}.npz').is_file(); need_tta=overwrite or not tdest.is_file()
  if not need_original and not need_tta: print(f'[{i}/{len(split)}] skip {cid}'); continue
  data,_,props=prep.run_case([str(x) for x in images],None,predictor.plans_manager,predictor.configuration_manager,predictor.dataset_json); tensor=torch.from_numpy(data).float().contiguous()
  predictions=original.parent/'predictions'; predictions.mkdir(parents=True,exist_ok=True)
  if need_original:
   predictor.use_mirroring=True
   logits=predictor.predict_logits_from_preprocessed_data(tensor).numpy(); (export_deferred(logits,props,original/cid) if model.get('defer_probability_canonicalization') else export_prediction_from_logits(logits,props,predictor.configuration_manager,predictor.plans_manager,predictor.dataset_json,str(original/cid),save_probabilities=True,num_threads_torch=1)); del logits
   # nnU-Net writes the original-space hard mask beside the probability NPZ.
   # Keep the NPZ in probabilities/ and place the serving/audit mask in predictions/.
   mask_source=original/f'{cid}.nii.gz'
   if not mask_source.is_file(): raise FileNotFoundError(mask_source)
   if not model.get('defer_probability_canonicalization'):
    canonicalize_probability(original/f'{cid}.npz',mask_source,float(model['threshold']))
   shutil.copy2(mask_source,predictions/f'{cid}.nii.gz')
  elif not (predictions/f'{cid}.nii.gz').is_file():
   mask_source=original/f'{cid}.nii.gz'
   if not mask_source.is_file(): raise FileNotFoundError(f'Missing existing mask for {cid}: {mask_source}')
   shutil.copy2(mask_source,predictions/f'{cid}.nii.gz')
  if need_tta:
   predictor.use_mirroring=False
   flipped=torch.flip(tensor,dims=[axis+1]); logits=predictor.predict_logits_from_preprocessed_data(flipped); logits=torch.flip(logits,dims=[axis+1]).numpy(); (export_deferred(logits,props,tta/cid) if model.get('defer_probability_canonicalization') else export_prediction_from_logits(logits,props,predictor.configuration_manager,predictor.plans_manager,predictor.dataset_json,str(tta/cid),save_probabilities=True,num_threads_torch=1)); del logits,flipped
   tta_mask=tta/f'{cid}.nii.gz'
   if not tta_mask.is_file(): raise FileNotFoundError(tta_mask)
   if not model.get('defer_probability_canonicalization'):
    canonicalize_probability(tta/f'{cid}.npz',tta_mask,float(model['threshold']))
  del tensor,data; torch.cuda.empty_cache(); print(f'[{i}/{len(split)}] {cid} preprocessed-axis={axis}',flush=True)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--model-id',required=True); ap.add_argument('--registry',type=Path,default=Path('configs/calibration/models.yaml')); ap.add_argument('--overwrite',action='store_true'); ap.add_argument('--normalize-existing',action='store_true',help='repair existing original-space NPZ orientation without GPU inference'); ap.add_argument('--limit',type=int); a=ap.parse_args(); repo=Path(__file__).resolve().parents[2]
 from brainon_ai.calibration.registry import resolve_model
 m=resolve_model(a.model_id,a.registry,repo); m['_limit']=a.limit
 if a.normalize_existing:
  normalize_existing_exports(m); return
 if not torch.cuda.is_available(): raise RuntimeError('CUDA required for TTA inference')
 (segresnet if m.get('backend')=='segresnet' else nnunet)(m,repo,a.overwrite)
if __name__=='__main__': main()
