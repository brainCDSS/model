#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from itertools import permutations
from pathlib import Path
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np,pandas as pd,yaml
from brainon_ai.calibration.features import foreground_probability,ensemble_disagreement
from brainon_ai.calibration.repeated_cv import run_repeated_cv
REPO=Path(__file__).resolve().parents[2]; ART=REPO/'artifacts/calibration_v1'
OLD={'H-T02-E300':REPO/'artifacts/calibration/hemorrhage/H-T02-E300/validation-best-a2dda949fd97/score_features.csv','I-E03':REPO/'artifacts/calibration/ischemia/I-E03/validation-best-795472c38a4e/score_features.csv'}
SHA={'H-T02-E300':'a2dda949fd977f6ac112bcf3192b6a5f22a107c66d7c9f2d19a59423e4918bba','I-E03':'795472c38a4e8f920f60de3e9cef02d60f35a447f3a1dfdb216d2f86118c31ed'}
TARGET={'H-T02-E300':'hemorrhagic_stroke_ct','I-E03':'ischemic_stroke_mri'}
def align(a,target,source=None,mask=None,threshold=.5):
    a=np.asarray(a,np.float32); shape=source.shape if source is not None else target.shape
    if a.shape!=shape:
        candidates=[a.transpose(x) for x in permutations(range(3)) if a.transpose(x).shape==shape]
        if not candidates: raise ValueError(f'{a.shape} cannot align to {shape}')
        if len(candidates)>1 and mask is not None: a=candidates[int(np.argmax([np.mean((x>=threshold)==mask) for x in candidates]))]
        elif len(candidates)==1:a=candidates[0]
        else:raise ValueError('ambiguous orientation without mask')
    if source is None or (source.shape==target.shape and np.allclose(source.affine,target.affine,atol=1e-4)):return a
    return np.clip(np.asanyarray(resample_from_to(nib.Nifti1Image(a,source.affine),target,order=1).dataobj),0,1).astype(np.float32)
def add_mri_disagreement(frame):
    final_dir=REPO/'runs/calibration_exports/I-E03/probabilities'; labels=REPO.parent/'datasets/ischemic/nnunet/nnUNet_raw/Dataset502_ISLES2022/labelsTr'; seglabels=REPO.parent/'datasets/ischemic/SegResNet/val/labels'
    nd=REPO/'runs/I-nnunet/Dataset502_ISLES2022/nnUNetTrainerIschemiaMultimodal__IschemiaPlansDualEncoderGatedV1__3d_fullres/fold_0/best_probability_validation'; s1=REPO/'runs/I-segresnet/I-S101/final_metrics/val_probabilities'; s5=REPO/'runs/I-segresnet/I-S50/final_metrics/val_probabilities'
    values=[]
    for cid in frame.case_id:
        ref=nib.load(str(labels/f'{cid}.nii.gz')); final=foreground_probability(final_dir/f'{cid}.npz'); final=align(final,ref)
        nim=nib.load(str(nd/f'{cid}.nii.gz')); n=foreground_probability(nd/f'{cid}.npz'); n=align(n,ref,nim,np.asanyarray(nim.dataobj)>0,.5)
        sim=nib.load(str(seglabels/f'{cid}_mask.nii.gz')); a=align(foreground_probability(s1/f'{cid}.npz'),ref,sim);b=align(foreground_probability(s5/f'{cid}.npz'),ref,sim)
        values.append(ensemble_disagreement([n,a,b],final,.20))
    frame=frame.copy();frame['ensemble_disagreement']=values;return frame
def specs(model):
    if model=='H-T02-E300':
        base=[{'name':'foreground_uncertainty','sign':-1},{'name':'tta_consistency','sign':1},{'name':'spatial_coherence','sign':1}]
        return {'CT-A-count-identity':base+[{'name':'predicted_component_count','sign':-1}], 'CT-A-count-log1p':base+[{'name':'predicted_component_count','transform':'log1p','sign':-1}], 'CT-B-count-identity':base+[{'name':'predicted_component_count','sign':-1},{'name':'component_probability_std','sign':-1}], 'CT-B-count-log1p':base+[{'name':'predicted_component_count','transform':'log1p','sign':-1},{'name':'component_probability_std','sign':-1}]}
    base=[{'name':'foreground_uncertainty','sign':-1},{'name':'tta_consistency','sign':1},{'name':'spatial_coherence','sign':1}]
    return {'MRI-A':base,'MRI-B':base+[{'name':'ensemble_disagreement','sign':-1}]}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model-id',choices=['H-T02-E300','I-E03'],required=True);a=ap.parse_args();frame=pd.read_csv(OLD[a.model_id]);
    if a.model_id=='I-E03':frame=add_mri_disagreement(frame)
    target=yaml.safe_load((REPO/f'configs/calibration/quality_targets/{TARGET[a.model_id]}.yaml').read_text()); summaries=[]
    for candidate,fs in specs(a.model_id).items():
        methods=[False,True] if a.model_id=='I-E03' else [False]
        for constrained in methods:
            name=candidate+('-constrained' if constrained else '-unconstrained');out=ART/a.model_id/name;artifact,stability=run_repeated_cv(frame,fs,out,a.model_id,SHA[a.model_id],target,constrained=constrained);m=artifact['metrics'];summaries.append({'candidate':name,'features':[x['name'] for x in fs],'transforms':[x.get('transform','identity') for x in fs],'method':artifact['method'],'cases':len(frame),'acceptable':int(frame.acceptable_segmentation.sum()),'unacceptable':int((1-frame.acceptable_segmentation).sum()),'cv_auroc':m['reliability_auroc'],'cv_auprc':m['reliability_auprc'],'brier':m['reliability_brier'],'ece':m['reliability_ece'],'mean_sign_stability':float(np.maximum(stability.positive_sign_ratio,stability.negative_sign_ratio).mean())})
    result=pd.DataFrame(summaries).sort_values(['cv_auroc','brier'],ascending=[False,True]); root=ART/a.model_id;result.to_csv(root/'candidate_comparison.csv',index=False);(root/'candidate_comparison.json').write_text(json.dumps(summaries,indent=2)+'\n');print(result.to_string(index=False))
if __name__=='__main__':main()
