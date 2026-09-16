#!/usr/bin/env python3
"""Average aligned foreground probability maps and select a lesion-F2 threshold."""
from __future__ import annotations
import argparse, json
from itertools import combinations
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import label

def fg(path):
 with np.load(path) as z:
  a=z['probabilities'] if 'probabilities' in z else z['softmax']
 return np.asarray(a[-1],dtype=np.float32)
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
 ap=argparse.ArgumentParser(); ap.add_argument('--model',action='append',required=True,help='NAME=/probability/dir'); ap.add_argument('--label-dir',required=True,type=Path); ap.add_argument('--output-dir',required=True,type=Path); ap.add_argument('--max-fp-per-scan',type=float,default=1.5); a=ap.parse_args()
 models=dict(item.split('=',1) for item in a.model); labels=sorted(a.label_dir.glob('*.nii.gz')); ids=[x.name.removesuffix('_mask.nii.gz').removesuffix('.nii.gz') for x in labels]; gts=[]
 for lp in labels:
  im=nib.load(lp); gts.append((np.asanyarray(im.dataobj)>0,float(np.prod(im.header.get_zooms()[:3]))/1000))
 arrays={name:[fg(Path(d)/f'{cid}.npz') for cid in ids] for name,d in models.items()}
 for name,vals in arrays.items():
  for cid,pv,(gt,_) in zip(ids,vals,gts):
   if pv.shape!=gt.shape: raise ValueError(f'{name}/{cid}: {pv.shape} != {gt.shape}')
 rows=[]
 candidates={**arrays,'soft_vote__'+'+'.join(arrays):[np.mean(x,axis=0) for x in zip(*arrays.values())]}
 for name,probs in candidates.items():
  for t in np.arange(.1,.91,.05): rows.append({'candidate':name,'threshold':round(float(t),2),**metrics(probs,gts,float(t))})
 df=pd.DataFrame(rows); a.output_dir.mkdir(parents=True,exist_ok=True); df.to_csv(a.output_dir/'ensemble_threshold_sweep.csv',index=False)
 ensemble=df[df.candidate.str.startswith('soft_vote__')]; eligible=ensemble[ensemble.fp_per_scan<=a.max_fp_per_scan]; selected=(eligible if len(eligible) else ensemble).sort_values(['lesion_f2','macro_dice_positive'],ascending=False).iloc[0].to_dict()
 diversity=[]
 for left,right in combinations(arrays,2):
  disagreements=[float(np.mean(np.abs(x-y))) for x,y in zip(arrays[left],arrays[right])]; diversity.append({'model_a':left,'model_b':right,'mean_probability_disagreement':float(np.mean(disagreements))})
 pd.DataFrame(diversity).to_csv(a.output_dir/'model_diversity.csv',index=False); (a.output_dir/'ensemble_summary.json').write_text(json.dumps({'models':{k:str(v) for k,v in models.items()},'selected':selected},ensure_ascii=False,indent=2)+'\n'); print(json.dumps(selected,ensure_ascii=False,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
