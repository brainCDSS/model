#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from itertools import permutations
from pathlib import Path
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np,pandas as pd,yaml
from sklearn.metrics import average_precision_score,brier_score_loss,confusion_matrix,f1_score,precision_score,recall_score,roc_auc_score
from brainon_ai.calibration.features import extract_features,foreground_probability,load_label,segmentation_metrics
from brainon_ai.calibration.pipeline import _ece,_label_path,_target
from brainon_ai.calibration.registry import resolve_model
from brainon_ai.calibration.scoring import calculate_prediction_score
TASK_CONFIG={'hemorrhage':'hemorrhagic_stroke_ct','ischemia':'ischemic_stroke_mri','aneurysm':'cerebral_aneurysm_mra'}
def aligned_probability(probability_path:Path,reference_path:Path,target:nib.Nifti1Image,threshold:float)->np.ndarray:
 p=foreground_probability(probability_path)
 if not reference_path.is_file():
  if p.shape != target.shape: raise FileNotFoundError(f'{reference_path} missing and probability {p.shape} != target {target.shape}')
  return np.asarray(p,np.float32)
 source=nib.load(str(reference_path))
 if p.shape!=source.shape:
  candidates=[p.transpose(order) for order in permutations(range(3)) if p.transpose(order).shape==source.shape]
  if not candidates:raise ValueError(f'{probability_path}: cannot align {p.shape} to {source.shape}')
  if len(candidates)>1:
   mask=np.asanyarray(source.dataobj)>0;agreement=[np.mean((x>=threshold)==mask) for x in candidates];p=candidates[int(np.argmax(agreement))]
  else:p=candidates[0]
 if source.shape==target.shape and np.allclose(source.affine,target.affine,atol=1e-4):return np.asarray(p,np.float32)
 image=nib.Nifti1Image(np.asarray(p,np.float32),source.affine,source.header)
 return np.clip(np.asanyarray(resample_from_to(image,(target.shape,target.affine),order=1).dataobj),0,1).astype(np.float32)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--export-model-id',required=True);ap.add_argument('--score-model',required=True,type=Path);ap.add_argument('--output-dir',required=True,type=Path);ap.add_argument('--registry',type=Path,default=Path('configs/calibration/models.yaml'));a=ap.parse_args()
 repo=Path(__file__).resolve().parents[2];m=resolve_model(a.export_model_id,a.registry,repo);score=json.loads(a.score_model.read_text());target=yaml.safe_load((repo/f"configs/calibration/quality_targets/{TASK_CONFIG[m['task']]}.yaml").read_text())
 rows=[]
 for pp in sorted(Path(m['prediction_dir']).glob('*.npz')):
  if pp.name.endswith('.normalizing.npz') or pp.name.endswith('.writing.npz'):continue
  cid=pp.stem;label_file=_label_path(Path(m['label_dir']),cid);target_image=nib.load(str(label_file));gt=np.asanyarray(target_image.dataobj)>0;root=Path(m['prediction_dir']).parent
  p=aligned_probability(pp,root/'predictions'/f'{cid}.nii.gz',target_image,float(m['threshold']))
  tta=[aligned_probability(Path(d)/f'{cid}.npz',Path(d)/f'{cid}.nii.gz',target_image,float(m['threshold'])) for d in m['tta_prediction_dirs']]
  feat=extract_features(p,float(m['threshold']),tta,int(m.get('connectivity',26)));quality=segmentation_metrics(p,gt,float(m['threshold']));prob=calculate_prediction_score(feat,score)/100
  rows.append({'case_id':cid,**feat,**quality,'acceptable_segmentation':_target(quality,target),'prediction_score':prob})
 frame=pd.DataFrame(rows);y=frame.acceptable_segmentation.to_numpy(int);prob=frame.prediction_score.to_numpy(float);threshold=float(score['threshold_selection']['selected_threshold']);pred=(prob>=threshold).astype(int);tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
 metrics={'case_count':len(frame),'acceptable_cases':int(y.sum()),'unacceptable_cases':int((1-y).sum()),'fixed_score_threshold':threshold,'reliability_auroc':float(roc_auc_score(y,prob)),'reliability_auprc':float(average_precision_score(y,prob)),'reliability_sensitivity':float(recall_score(y,pred,zero_division=0)),'reliability_specificity':float(recall_score(1-y,1-pred,zero_division=0)),'reliability_f1':float(f1_score(y,pred,zero_division=0)),'reliability_ppv':float(precision_score(y,pred,zero_division=0)),'reliability_npv':float(precision_score(1-y,1-pred,zero_division=0)),'reliability_brier':float(brier_score_loss(y,prob)),'reliability_ece':_ece(y,prob),'confusion_matrix':{'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)},'score_model':str(a.score_model),'refit_performed':False,'threshold_reselected':False}
 a.output_dir.mkdir(parents=True,exist_ok=True);frame.to_csv(a.output_dir/'external_score_predictions.csv',index=False);(a.output_dir/'external_score_evaluation.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n');print(json.dumps(metrics,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
