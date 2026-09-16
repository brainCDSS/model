from __future__ import annotations
import hashlib,json,subprocess,warnings
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss,log_loss,roc_auc_score
from sklearn.model_selection import StratifiedKFold,cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .features import FEATURE_NAMES,extract_features,foreground_probability,load_label,segmentation_metrics
from .scoring import calculate_prediction_score

def _case_id(path:Path)->str: return path.name.removesuffix('.npz')
def _label_path(label_dir:Path,cid:str)->Path:
    for name in (f"{cid}.nii.gz",f"{cid}_mask.nii.gz",f"{cid}.nii"):
        p=label_dir/name
        if p.is_file(): return p
    raise FileNotFoundError(f"label not found for {cid} in {label_dir}")
def _target(metric,cfg):
    if metric["gt_positive"]:
        return int(metric["dice"]>=float(cfg["dice_min"]) and metric["lesion_recall"]>=float(cfg["lesion_recall_min"]) and metric["fp_count"]<=int(cfg["fp_components_max"]))
    return int(metric["fp_count"]<=int(cfg["negative_fp_components_max"]))
def _ece(y,p,bins=10):
    edges=np.linspace(0,1,bins+1); out=0.
    for lo,hi in zip(edges[:-1],edges[1:]):
        sel=(p>=lo)&((p<hi) if hi<1 else (p<=hi))
        if sel.any(): out+=sel.mean()*abs(float(p[sel].mean())-float(y[sel].mean()))
    return float(out)
def fit_calibration(model:dict,target_cfg:dict,output_dir:Path,allow_provisional:bool=False,random_state:int=42)->dict:
    if target_cfg.get("status")!="approved" and not allow_provisional: raise RuntimeError("target thresholds are provisional; review config or pass --allow-provisional-target")
    pred_dir=Path(model["prediction_dir"]); label_dir=Path(model["label_dir"]); tta_dirs=[Path(x) for x in model.get("tta_prediction_dirs",[])]
    if not tta_dirs: raise RuntimeError("No actual TTA probability directory configured; tta_consistency will not be fabricated")
    rows=[]
    for pp in sorted(pred_dir.glob('*.npz')):
        cid=_case_id(pp); prob=foreground_probability(pp); lp=_label_path(label_dir,cid); gt,spatial=load_label(lp)
        if prob.shape!=gt.shape: raise ValueError(f"{cid}: probability {prob.shape} != label {gt.shape}")
        tt=[]
        for d in tta_dirs:
            q=d/f"{cid}.npz"
            if not q.is_file(): raise FileNotFoundError(q)
            tt.append(foreground_probability(q))
        feature=extract_features(prob,float(model["threshold"]),tt,int(model.get("connectivity",26))); metric=segmentation_metrics(prob,gt,float(model["threshold"]))
        rows.append({"case_id":cid,**feature,**metric,"acceptable_target":_target(metric,target_cfg),"label_path":str(lp),"voxel_volume_ml":float(np.prod(spatial['spacing_mm']))/1000})
    frame=pd.DataFrame(rows)
    if len(frame)<10: raise RuntimeError(f"too few calibration cases: {len(frame)}")
    X=frame[list(FEATURE_NAMES)].to_numpy(float); y=frame.acceptable_target.to_numpy(int); counts=np.bincount(y,minlength=2)
    if min(counts)<2: raise RuntimeError(f"target needs both classes with >=2 cases; counts={counts.tolist()}")
    folds=min(5,int(min(counts))); cv=StratifiedKFold(folds,shuffle=True,random_state=random_state)
    estimator=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced',random_state=random_state))
    oof=cross_val_predict(estimator,X,y,cv=cv,method='predict_proba')[:,1]; estimator.fit(X,y); scaler=estimator.named_steps['standardscaler']; lr=estimator.named_steps['logisticregression']
    near_zero=np.asarray(scaler.var_)<1e-12
    if near_zero.any(): warnings.warn(f"near-zero variance features: {[FEATURE_NAMES[i] for i in np.flatnonzero(near_zero)]}")
    artifact={"schema_version":"1.0","method":"logistic_regression","score":{"name":"ai_prediction_score","definition":"probability_of_acceptable_prediction","range":[0,100],"activation":"sigmoid"},"model":{"id":model['model_id'],"name":model.get('name',model['model_id']),"version":str(model['version']),"task":model['task'],"checkpoint_sha256":model.get("checkpoint_sha256")},"calibration_version":"1.0.0","feature_schema_version":"1.0","features":[{"name":n,"mean":float(scaler.mean_[i]),"std":float(scaler.scale_[i]),"coefficient":float(lr.coef_[0,i])} for i,n in enumerate(FEATURE_NAMES)],"intercept":float(lr.intercept_[0]),"empty_prediction_policy":{"foreground_top10_mean":0.0,"foreground_uncertainty":0.0,"spatial_coherence":0.0,"tta_consistency":"Dice convention: empty-empty=1"},"connectivity":int(model.get('connectivity',26)),"prediction_threshold":float(model['threshold']),"tta":{"enabled":True,"transforms":model.get('tta_transforms',[]),"probability_dirs":[str(x) for x in tta_dirs]},"target_definition":target_cfg,"data_leakage_control":{"fit_source":"validation/calibration predictions only","evaluation":"stratified out-of-fold predictions","cv_folds":folds,"test_set_used":False},"training":{"num_cases":len(frame),"num_acceptable":int(y.sum()),"num_unacceptable":int((1-y).sum())},"metrics":{"brier_score":float(brier_score_loss(y,oof)),"log_loss":float(log_loss(y,oof,labels=[0,1])),"ece":_ece(y,oof),"auroc":float(roc_auc_score(y,oof))},"created_at":datetime.now(timezone.utc).isoformat(),"git_commit":subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip() or None}
    frame['raw_logit']=np.log(np.clip(oof,1e-12,1-1e-12)/np.clip(1-oof,1e-12,1)); frame['calibrated_probability']=oof; frame['prediction_score']=100*oof
    reference=np.array([calculate_prediction_score(dict(zip(FEATURE_NAMES,row)),artifact) for row in X])/100
    if not np.allclose(reference,estimator.predict_proba(X)[:,1],atol=1e-10): raise AssertionError('JSON reference != sklearn full-fit prediction')
    output_dir.mkdir(parents=True,exist_ok=True); (output_dir/'calibration.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2)+'\n'); frame.to_csv(output_dir/'calibration_predictions.csv',index=False)
    curve=[]; edges=np.linspace(0,1,min(10,len(y))+1)
    for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        selected=(oof>=lo)&((oof<=hi) if i==len(edges)-2 else (oof<hi))
        if selected.any():
            curve.append({'bin_lower':float(lo),'bin_upper':float(hi),'predicted_probability':float(oof[selected].mean()),'observed_success_rate':float(y[selected].mean()),'case_count':int(selected.sum())})
    if sum(row['case_count'] for row in curve)!=len(y): raise AssertionError('calibration curve case count mismatch')
    pd.DataFrame(curve).to_csv(output_dir/'calibration_curve.csv',index=False); (output_dir/'calibration_metrics.json').write_text(json.dumps(artifact['metrics'],indent=2)+'\n')
    manifest={'model_id':model['model_id'],'model_version':str(model['version']),'files':{}}
    for name in ('calibration.json','calibration_metrics.json','calibration_predictions.csv','calibration_curve.csv'):
        manifest['files'][name]=hashlib.sha256((output_dir/name).read_bytes()).hexdigest()
    (output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n'); return artifact
