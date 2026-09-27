from __future__ import annotations
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy.ndimage import label

BASELINE_FEATURE_NAMES=(
    "foreground_top10_mean",
    "foreground_uncertainty",
    "tta_consistency",
    "spatial_coherence",
    "component_probability_std",
    "predicted_component_count",
)
# Backward-compatible public name. New callers should use BASELINE_FEATURE_NAMES.
FEATURE_NAMES=BASELINE_FEATURE_NAMES
STRUCTURE_26=np.ones((3,3,3),dtype=np.uint8)

def foreground_probability(path: str|Path)->np.ndarray:
    with np.load(path) as z:
        key="foreground" if "foreground" in z else "probabilities" if "probabilities" in z else "softmax" if "softmax" in z else None
        if key is None: raise KeyError(f"{path}: foreground/probabilities/softmax key missing")
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
    if fg.size==0:
        # Empty prediction semantics are part of the v1 serving contract. Count=0
        # is meaningful; foreground-derived statistics are represented by 0.
        top10=uncertainty=coherence=component_std=0.0; component_count=0
    else:
        k=max(1,int(np.ceil(fg.size*.10))); top10=float(np.mean(np.partition(fg,fg.size-k)[-k:]))
        clipped=np.clip(fg,1e-7,1-1e-7); uncertainty=float(np.mean(-clipped*np.log(clipped)-(1-clipped)*np.log(1-clipped)))
        cc,n=label(mask,structure=STRUCTURE_26); sizes=np.bincount(cc.ravel())[1:]; coherence=float(sizes.max()/fg.size) if n else 0.0
        component_count=int(n)
        component_means=np.asarray([p[cc==i].mean() for i in range(1,n+1)],dtype=float)
        component_std=float(component_means.std(ddof=0)) if component_means.size else 0.0
    consistencies=[]
    for tta in tta_probabilities:
        t=np.asarray(tta); 
        if t.shape!=p.shape: raise ValueError(f"TTA shape {t.shape} != original {p.shape}")
        consistencies.append(dice(mask,t>=threshold))
    if not consistencies: raise ValueError("tta_consistency requires at least one inverse-transformed TTA probability map")
    return {
        "foreground_top10_mean":top10,
        "foreground_uncertainty":uncertainty,
        "tta_consistency":float(np.mean(consistencies)),
        "spatial_coherence":coherence,
        "component_probability_std":component_std,
        "predicted_component_count":component_count,
    }

def segmentation_metrics(probability, ground_truth, threshold:float)->dict:
    pred=np.asarray(probability)>=threshold; gt=np.asarray(ground_truth)>0
    tp=int(np.logical_and(pred,gt).sum()); fpv=int(np.logical_and(pred,~gt).sum()); fn=int(np.logical_and(~pred,gt).sum())
    d=1.0 if 2*tp+fpv+fn==0 else 2*tp/(2*tp+fpv+fn)
    gc,gn=label(gt,structure=STRUCTURE_26); pc,pn=label(pred,structure=STRUCTURE_26)
    detected=sum(bool(pred[gc==i].any()) for i in range(1,gn+1)); fp=sum(not bool(gt[pc==i].any()) for i in range(1,pn+1))
    return {"dice":float(d),"lesion_recall":float(detected/gn) if gn else 1.0,"fp_count":int(fp),"gt_lesion_count":int(gn),"pred_lesion_count":int(pn),"gt_positive":bool(gn)}

def load_label(path:str|Path)->tuple[np.ndarray,dict]:
    im=nib.load(str(path)); return np.asanyarray(im.dataobj)>0,{"spacing_mm":[float(x) for x in im.header.get_zooms()[:3]],"affine":im.affine.tolist()}


def ensemble_disagreement(component_probabilities, final_probability, threshold: float) -> float:
    """Mean component-model probability SD inside the final predicted foreground."""
    arrays=[np.asarray(x,dtype=np.float32) for x in component_probabilities]
    final=np.asarray(final_probability,dtype=np.float32)
    if len(arrays)<2: raise ValueError("ensemble_disagreement requires at least two component probabilities")
    if any(x.shape!=final.shape for x in arrays): raise ValueError("component probability shape mismatch")
    mask=final>=float(threshold)
    if not mask.any(): return 0.0
    return float(np.std(np.stack(arrays,axis=0),axis=0,ddof=0)[mask].mean())

def component_reliability_features(probability, threshold: float, voxel_volume_ml: float,
                                   small_volume_ml: float=0.01, low_probability: float=0.75,
                                   rule: str="or", largest_mode: str="mean") -> dict:
    """MRA component features using fixed physical-volume and confidence rules."""
    p=np.asarray(probability,dtype=np.float32); mask=p>=float(threshold); cc,n=label(mask,structure=STRUCTURE_26)
    if n==0: return {"largest_component_confidence":0.0,"fp_like_component_count":0}
    parts=[]
    for i in range(1,n+1):
        values=p[cc==i]; volume=float(values.size)*float(voxel_volume_ml)
        parts.append((values,volume,float(values.mean())))
    largest=max(parts,key=lambda x:x[1])
    if largest_mode=="mean": confidence=largest[2]
    elif largest_mode=="top10_mean":
        values=largest[0]; k=max(1,int(np.ceil(values.size*.10))); confidence=float(np.mean(np.partition(values,values.size-k)[-k:]))
    else: raise ValueError(f"unsupported largest_mode: {largest_mode}")
    flags=[]
    for _,volume,mean_probability in parts:
        small=volume<float(small_volume_ml); low=mean_probability<float(low_probability)
        flags.append((small or low) if rule=="or" else (small and low) if rule=="and" else (_ for _ in ()).throw(ValueError(f"unsupported rule: {rule}")))
    return {"largest_component_confidence":float(confidence),"fp_like_component_count":int(sum(flags))}
