#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np,pandas as pd,yaml
from brainon_ai.calibration.features import foreground_probability,extract_features,segmentation_metrics,load_label,component_reliability_features
from brainon_ai.calibration.pipeline import _label_path,_target
from brainon_ai.calibration.repeated_cv import run_repeated_cv
REPO=Path(__file__).resolve().parents[2];MODEL='A-B04';ROOT=REPO/'runs/calibration_exports'/MODEL;LABELS=REPO.parent/'datasets/aneurysm/v2/model_ready/nnunet/nnUNet_raw/Dataset503_TopAneuMRABinary/labelsTr';OUT=REPO/'artifacts/calibration_v1'/MODEL
def main():
 probs=ROOT/'probabilities';tta=ROOT/'tta_probabilities/axis0_flip'
 if not probs.is_dir() or len(list(probs.glob('*.npz')))==0: raise RuntimeError('A-B04 calibration probability/TTA export missing. Run prepare_final_calibration_exports.py --model-id A-B04 --run-gpu-export first.')
 target=yaml.safe_load((REPO/'configs/calibration/quality_targets/cerebral_aneurysm_mra.yaml').read_text());base=[];cache={}
 for pp in sorted(probs.glob('*.npz')):
  cid=pp.stem;p=foreground_probability(pp);tp=foreground_probability(tta/f'{cid}.npz');lp=_label_path(LABELS,cid);gt,meta=load_label(lp);f=extract_features(p,.5,[tp]);m=segmentation_metrics(p,gt,.5);base.append({'case_id':cid,**f,**m,'acceptable_segmentation':_target(m,target),'voxel_volume_ml':float(np.prod(meta['spacing_mm']))/1000});cache[cid]=p
 base=pd.DataFrame(base);summaries=[]
 for mode in ['mean','top10_mean']:
  for volume in [.005,.01,.02]:
   for probability in [.65,.75]:
    for rule in ['and','or']:
     frame=base.copy();vals=[component_reliability_features(cache[c],.5,v,volume,probability,rule,mode) for c,v in zip(frame.case_id,frame.voxel_volume_ml)];frame['largest_component_confidence']=[x['largest_component_confidence'] for x in vals];frame['fp_like_component_count']=[x['fp_like_component_count'] for x in vals]
     name=f'{mode}-vol{volume:g}-p{probability:g}-{rule}';spec=[{'name':'foreground_uncertainty','sign':-1},{'name':'tta_consistency','sign':1},{'name':'largest_component_confidence','sign':1},{'name':'fp_like_component_count','transform':'log1p','sign':-1}];artifact,stability=run_repeated_cv(frame,spec,OUT/name,MODEL,None,target,constrained=False);m=artifact['metrics'];summaries.append({'candidate':name,'largest_component_confidence':mode,'small_volume_ml':volume,'low_probability':probability,'rule':rule,'cv_auroc':m['reliability_auroc'],'cv_auprc':m['reliability_auprc'],'brier':m['reliability_brier'],'ece':m['reliability_ece'],'mean_sign_stability':float(np.maximum(stability.positive_sign_ratio,stability.negative_sign_ratio).mean())})
 result=pd.DataFrame(summaries).sort_values(['cv_auroc','brier'],ascending=[False,True]);result.to_csv(OUT/'candidate_comparison.csv',index=False);(OUT/'candidate_comparison.json').write_text(json.dumps(summaries,indent=2)+'\n');print(result.head(12).to_string(index=False))
if __name__=='__main__':main()
