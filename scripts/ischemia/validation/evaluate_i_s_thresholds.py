#!/usr/bin/env python3
"""Threshold sweep with voxel, lesion, small-lesion and negative-case metrics."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import label
from brainon_ai.models.ischemia.segresnet_hy.metrics import calculate_lesion_metrics, calculate_voxel_metrics

def components(x): return label(x, structure=np.ones((3,3,3),dtype=np.uint8))
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--data-root',required=True,type=Path); ap.add_argument('--probability-dir',required=True,type=Path); ap.add_argument('--output-dir',required=True,type=Path); ap.add_argument('--max-fp-per-scan',type=float,default=1.5); ap.add_argument('--small-lesion-max-ml',type=float,default=1.0); a=ap.parse_args()
 labels=sorted((a.data_root/'val/labels').glob('*_mask.nii.gz')); cases=[]
 for lp in labels:
  cid=lp.name.removesuffix('_mask.nii.gz'); img=nib.load(lp); gt=np.asanyarray(img.dataobj)>0; prob=np.load(a.probability_dir/f'{cid}.npz')['probabilities'][-1]; cc,n=components(gt); vv=float(np.prod(img.header.get_zooms()[:3]))/1000
  small=[i for i in range(1,n+1) if np.sum(cc==i)*vv<a.small_lesion_max_ml]; cases.append((cid,prob,gt,cc,n,small))
 rows=[]
 for t in np.arange(.1,.91,.05):
  ds=[]; vr=[]; vp=[]; ng=nd=nt=nfp=sm=smd=0; small_dice=[]; neg=negfp=0
  for cid,prob,gt,cc,n,small in cases:
   pred=prob>=t; v=calculate_voxel_metrics(pred,gt); l=calculate_lesion_metrics(pred,gt); pc,pn=components(pred)
   if gt.any(): ds.append(v['dice']); vr.append(v['recall_sensitivity']); vp.append(v['precision'])
   else: neg+=1; negfp+=int(pred.any())
   ng+=n; nd+=l['detected_gt_lesions']; nt+=l['pred_lesion_count']; nfp+=l['false_positive_lesions']; sm+=len(small); smd+=sum(bool(pred[cc==i].any()) for i in small)
   for i in small:
    gm=cc==i; overlap_ids=np.unique(pc[gm]); overlap_ids=overlap_ids[overlap_ids>0]; pm=np.isin(pc,overlap_ids) if len(overlap_ids) else np.zeros_like(pred); den=int(gm.sum()+pm.sum()); small_dice.append(2*int(np.logical_and(gm,pm).sum())/den if den else 0.0)
  rec=nd/ng if ng else 1.; prec=(nt-nfp)/nt if nt else 1.; f1=2*prec*rec/(prec+rec) if prec+rec else 0.; f2=5*prec*rec/(4*prec+rec) if 4*prec+rec else 0.
  rows.append(dict(threshold=round(float(t),2),macro_dice_positive=float(np.mean(ds)),voxel_recall=float(np.mean(vr)),voxel_precision=float(np.mean(vp)),lesion_recall=rec,lesion_precision=prec,lesion_f1=f1,lesion_f2=f2,small_lesion_recall=smd/sm if sm else None,small_lesion_dice=float(np.mean(small_dice)) if small_dice else None,fp_per_scan=nfp/len(cases),negative_case_fp_rate=negfp/neg if neg else None,detected_lesions=nd,total_gt_lesions=ng,false_positive_lesions=nfp))
 a.output_dir.mkdir(parents=True,exist_ok=True); df=pd.DataFrame(rows); df.to_csv(a.output_dir/'threshold_sweep.csv',index=False); eligible=df[df.fp_per_scan<=a.max_fp_per_scan]; pool=eligible if len(eligible) else df; rec=pool.sort_values(['lesion_f2','macro_dice_positive'],ascending=False).iloc[0].to_dict(); best=df.sort_values('macro_dice_positive',ascending=False).iloc[0].to_dict(); payload={'small_lesion_max_ml':a.small_lesion_max_ml,'max_fp_per_scan':a.max_fp_per_scan,'recommended_lesion_f2':rec,'best_macro_dice':best}; (a.output_dir/'threshold_summary.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(payload,ensure_ascii=False,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
