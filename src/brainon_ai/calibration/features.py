from __future__ import annotations
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy.ndimage import label

FEATURE_NAMES=("foreground_top10_mean","foreground_uncertainty","spatial_coherence","tta_consistency")
STRUCTURE_26=np.ones((3,3,3),dtype=np.uint8)

def foreground_probability(path: str|Path)->np.ndarray:
    with np.load(path) as z:
        key="probabilities" if "probabilities" in z else "softmax" if "softmax" in z else None
        if key is None: raise KeyError(f"{path}: probabilities/softmax key missing")
        a=np.asarray(z[key])
    if a.ndim==4: a=a[-1]
    if a.ndim!=3: raise ValueError(f"{path}: expected 3D or [C,3D], got {a.shape}")
    if not np.isfinite(a).all(): raise ValueError(f"{path}: non-finite probability")
    return a.astype(np.float32,copy=False)

def dice(a,b)->float:
    a=np.asarray(a,bool); b=np.asarray(b,bool); den=int(a.sum()+b.sum())
    return 1.0 if den==0 else 2.0*float(np.logical_and(a,b).sum())/den

def extract_features(probability, threshold:float, tta_probabilities=(), connectivity:int=26)->dict:
    if connectivity!=26: raise ValueError("v1 supports 26-connectivity only")
    p=np.asarray(probability,dtype=np.float32); mask=p>=float(threshold); fg=p[mask]
    if fg.size==0: top10=uncertainty=coherence=0.0
    else:
        k=max(1,int(np.ceil(fg.size*.10))); top10=float(np.mean(np.partition(fg,fg.size-k)[-k:]))
        clipped=np.clip(fg,1e-7,1-1e-7); uncertainty=float(np.mean(-clipped*np.log(clipped)-(1-clipped)*np.log(1-clipped)))
        cc,n=label(mask,structure=STRUCTURE_26); sizes=np.bincount(cc.ravel())[1:]; coherence=float(sizes.max()/fg.size) if n else 0.0
    consistencies=[]
    for tta in tta_probabilities:
        t=np.asarray(tta); 
        if t.shape!=p.shape: raise ValueError(f"TTA shape {t.shape} != original {p.shape}")
        consistencies.append(dice(mask,t>=threshold))
    if not consistencies: raise ValueError("tta_consistency requires at least one inverse-transformed TTA probability map")
    return {"foreground_top10_mean":top10,"foreground_uncertainty":uncertainty,"spatial_coherence":coherence,"tta_consistency":float(np.mean(consistencies))}

def segmentation_metrics(probability, ground_truth, threshold:float)->dict:
    pred=np.asarray(probability)>=threshold; gt=np.asarray(ground_truth)>0
    tp=int(np.logical_and(pred,gt).sum()); fpv=int(np.logical_and(pred,~gt).sum()); fn=int(np.logical_and(~pred,gt).sum())
    d=1.0 if 2*tp+fpv+fn==0 else 2*tp/(2*tp+fpv+fn)
    gc,gn=label(gt,structure=STRUCTURE_26); pc,pn=label(pred,structure=STRUCTURE_26)
    detected=sum(bool(pred[gc==i].any()) for i in range(1,gn+1)); fp=sum(not bool(gt[pc==i].any()) for i in range(1,pn+1))
    return {"dice":float(d),"lesion_recall":float(detected/gn) if gn else 1.0,"fp_count":int(fp),"gt_lesion_count":int(gn),"pred_lesion_count":int(pn),"gt_positive":bool(gn)}

def load_label(path:str|Path)->tuple[np.ndarray,dict]:
    im=nib.load(str(path)); return np.asanyarray(im.dataobj)>0,{"spacing_mm":[float(x) for x in im.header.get_zooms()[:3]],"affine":im.affine.tolist()}
