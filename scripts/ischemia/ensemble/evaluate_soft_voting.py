#!/usr/bin/env python3
"""Average aligned foreground probability maps and select a lesion-F2 threshold."""
from __future__ import annotations
import argparse, json
from itertools import combinations, permutations
from pathlib import Path
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import pandas as pd
from scipy.ndimage import label

def fg(path):
 with np.load(path) as z:
  a=z['probabilities'] if 'probabilities' in z else z['softmax']
 return np.asarray(a[-1],dtype=np.float32)
def align_probability(prob, target_image, prediction_file=None):
 # First restore archive axes to the NIfTI exported with that model, then
 # resample in physical space to the shared target grid when required.
 source_image=nib.load(prediction_file) if prediction_file is not None and prediction_file.is_file() else None
 source_shape=source_image.shape if source_image is not None else target_image.shape
 if prob.shape != source_shape:
  candidates=[prob.transpose(axis) for axis in permutations(range(3)) if prob.transpose(axis).shape == source_shape]
  if not candidates: raise ValueError(f'cannot align probability {prob.shape} to source {source_shape}')
  if len(candidates)==1: prob=candidates[0]
  else:
   saved=np.asanyarray(source_image.dataobj)>0
   agreement=[np.mean((x>=.5)==saved) for x in candidates]
   best=int(np.argmax(agreement))
   if agreement[best]<.99999: raise ValueError(f'could not verify axes: best agreement={agreement[best]:.6f}')
   prob=candidates[best]
 if source_image is None:
  return prob.astype(np.float32,copy=False)
 if source_image.shape == target_image.shape and np.allclose(source_image.affine,target_image.affine,atol=1e-4):
  return prob.astype(np.float32,copy=False)
 source_probability=nib.Nifti1Image(prob.astype(np.float32,copy=False),source_image.affine,source_image.header)
 return np.clip(np.asanyarray(resample_from_to(source_probability,(target_image.shape,target_image.affine),order=1).dataobj),0,1).astype(np.float32,copy=False)
def cc(x): return label(x,structure=np.ones((3,3,3),np.uint8))
def metrics(probs,gts,threshold):
 dice=[]; vrec=[]; vprec=[]; total_gt=det=total_pred=fp=0; small=small_det=0; neg=neg_fp=0
 for prob,(gt,vv) in zip(probs,gts):
  pred=prob>=threshold; tp=np.logical_and(pred,gt).sum(); f=np.logical_and(pred,~gt).sum(); fn=np.logical_and(~pred,gt).sum()
  if gt.any(): dice.append(2*tp/(2*tp+f+fn) if 2*tp+f+fn else 1); vrec.append(tp/(tp+fn) if tp+fn else 1); vprec.append(tp/(tp+f) if tp+f else 1)
  else: neg+=1; neg_fp+=int(pred.any())
  gc,gn=cc(gt); pc,pn=cc(pred); total_gt+=gn; total_pred+=pn
  det+=sum(bool(pred[gc==i].any()) for i in range(1,gn+1)); fp+=sum(not bool(gt[pc==i].any()) for i in range(1,pn+1))
  ids=[i for i in range(1,gn+1) if (gc==i).sum()*vv<1.0]; small+=len(ids); small_det+=sum(bool(pred[gc==i].any()) for i in ids)
 rec=det/total_gt if total_gt else 1.; prec=(total_pred-fp)/total_pred if total_pred else 1.; f1=2*prec*rec/(prec+rec) if prec+rec else 0.; f2=5*prec*rec/(4*prec+rec) if 4*prec+rec else 0.
 return dict(macro_dice_positive=float(np.mean(dice)),voxel_recall=float(np.mean(vrec)),voxel_precision=float(np.mean(vprec)),lesion_recall=rec,lesion_precision=prec,lesion_f1=f1,lesion_f2=f2,small_lesion_recall=small_det/small if small else None,fp_per_scan=fp/len(gts),negative_case_fp_rate=neg_fp/neg if neg else None)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--model',action='append',required=True,help='NAME=/probability/dir or NAME=/probability/dir::/reference/nifti/dir'); ap.add_argument('--weight',action='append',type=float,help='Model weights in --model order; defaults to equal'); ap.add_argument('--label-dir',required=True,type=Path); ap.add_argument('--output-dir',required=True,type=Path); ap.add_argument('--max-fp-per-scan',type=float,default=1.5); a=ap.parse_args()
 model_specs={}
 for item in a.model:
  name,spec=item.split('=',1)
  # NAME=PROBABILITY_DIR or NAME=PROBABILITY_DIR::REFERENCE_NIFTI_DIR
  probability_dir,separator,reference_dir=spec.partition('::')
  model_specs[name]=(Path(probability_dir),Path(reference_dir) if separator else Path(probability_dir))
 # Evaluate only the exact case intersection exported by every model.
 labels=[lp for lp in sorted(a.label_dir.glob('*.nii.gz')) if all((probability_dir/f'{lp.name.removesuffix('_mask.nii.gz').removesuffix('.nii.gz')}.npz').is_file() for probability_dir,_ in model_specs.values())]
 if not labels: raise FileNotFoundError('No common label/probability cases found')
 ids=[x.name.removesuffix('_mask.nii.gz').removesuffix('.nii.gz') for x in labels]; target_images=[]; gts=[]
 for lp in labels:
  image=nib.load(lp); target_images.append(image); gts.append((np.asanyarray(image.dataobj)>0,float(np.prod(image.header.get_zooms()[:3]))/1000))
 arrays={}
 for name,(probability_dir,reference_dir) in model_specs.items():
  values=[]
  for cid,target_image in zip(ids,target_images):
   probability=fg(probability_dir/f'{cid}.npz')
   reference=reference_dir/f'{cid}.nii.gz'
   if not reference.is_file(): reference=reference_dir/f'{cid}_pred_mask.nii.gz'
   values.append(align_probability(probability, target_image, reference if reference.is_file() else None))
  arrays[name]=values
 models={name:str(probability_dir) for name,(probability_dir,_) in model_specs.items()}
 weights=np.asarray(a.weight if a.weight is not None else [1.0]*len(arrays),dtype=np.float32)
 if len(weights)!=len(arrays) or np.any(weights<0) or weights.sum()<=0: raise ValueError('--weight must contain one nonnegative value per --model and sum to > 0')
 weights=weights/weights.sum()
 rows=[]
 candidate_name='soft_vote__'+'+'.join(f'{name}={weight:.2f}' for name,weight in zip(arrays,weights))
 candidates={**arrays,candidate_name:[np.average(np.stack(x),axis=0,weights=weights).astype(np.float32) for x in zip(*arrays.values())]}
 for name,probs in candidates.items():
  for t in np.arange(.1,.91,.05): rows.append({'candidate':name,'threshold':round(float(t),2),**metrics(probs,gts,float(t))})
 df=pd.DataFrame(rows); a.output_dir.mkdir(parents=True,exist_ok=True); df.to_csv(a.output_dir/'ensemble_threshold_sweep.csv',index=False)
 ensemble=df[df.candidate.str.startswith('soft_vote__')]; eligible=ensemble[ensemble.fp_per_scan<=a.max_fp_per_scan]; selected=(eligible if len(eligible) else ensemble).sort_values(['lesion_f2','macro_dice_positive'],ascending=False).iloc[0].to_dict()
 probability_dir=a.output_dir/'probabilities'; probability_dir.mkdir(exist_ok=True)
 for cid,probability,label_path in zip(ids,candidates[candidate_name],labels):
  np.savez_compressed(probability_dir/f'{cid}.npz',probabilities=np.stack([1-probability,probability]))
  image=nib.load(label_path); nib.save(nib.Nifti1Image((probability>=selected['threshold']).astype(np.uint8),image.affine,image.header),probability_dir/f'{cid}.nii.gz')
 diversity=[]
 for left,right in combinations(arrays,2):
  disagreements=[float(np.mean(np.abs(x-y))) for x,y in zip(arrays[left],arrays[right])]; diversity.append({'model_a':left,'model_b':right,'mean_probability_disagreement':float(np.mean(disagreements))})
 pd.DataFrame(diversity).to_csv(a.output_dir/'model_diversity.csv',index=False); (a.output_dir/'ensemble_summary.json').write_text(json.dumps({'models':{k:str(v) for k,v in models.items()},'weights':{name:float(weight) for name,weight in zip(arrays,weights)}, 'probability_dir':str(probability_dir),'selected':selected},ensure_ascii=False,indent=2)+'\n'); print(json.dumps(selected,ensure_ascii=False,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
