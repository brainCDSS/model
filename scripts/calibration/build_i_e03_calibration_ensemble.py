#!/usr/bin/env python3
from __future__ import annotations
from itertools import permutations
from pathlib import Path
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
from brainon_ai.calibration.features import foreground_probability

REPO=Path(__file__).resolve().parents[2]
ORIGINAL=REPO/'runs/ischemia_ensemble/I-E03_N120_S50_S101/N120_060_S101_020_S50_020/probabilities'
LABELS=REPO.parent/'datasets/ischemic/nnunet/nnUNet_raw/Dataset502_ISLES2022/labelsTr'
N120=REPO/'runs/calibration_components/I-E03/I-N120/tta_probabilities/axis0_flip'
S101=REPO/'runs/calibration_components/I-E03/I-S101/tta_probabilities/axis0_flip'
S50=REPO/'runs/calibration_tta/I-S50/axis0_flip'
SEG_LABELS=REPO.parent/'datasets/ischemic/SegResNet/val/labels'
OUT=REPO/'runs/calibration_exports/I-E03'
THRESHOLD=.20

def align(array:np.ndarray,target:nib.Nifti1Image,name:str,source:nib.Nifti1Image|None=None)->np.ndarray:
 source_shape=source.shape if source is not None else target.shape
 if array.shape!=source_shape:
  candidates=[array.transpose(order) for order in permutations(range(3)) if array.transpose(order).shape==source_shape]
  if len(candidates)!=1:raise ValueError(f'{name}: shape {array.shape} cannot be uniquely aligned to source {source_shape}; candidates={len(candidates)}')
  array=candidates[0]
 if source is None or (source.shape==target.shape and np.allclose(source.affine,target.affine,atol=1e-4)):return np.asarray(array,np.float32)
 image=nib.Nifti1Image(np.asarray(array,np.float32),source.affine,source.header)
 return np.clip(np.asanyarray(resample_from_to(image,(target.shape,target.affine),order=1).dataobj),0,1).astype(np.float32)

def save(path:Path,array:np.ndarray)->None:
 path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_name(path.stem+'.writing.npz')
 np.savez_compressed(temp,foreground=np.asarray(array,np.float32));temp.replace(path)

def main()->int:
 cases=sorted(x.stem for x in ORIGINAL.glob('*.npz'))
 if len(cases)!=38:raise RuntimeError(f'expected 38 I-E03 cases, got {len(cases)}')
 for i,cid in enumerate(cases,1):
  label=LABELS/f'{cid}.nii.gz'; ref=nib.load(str(label)); shape=tuple(ref.shape)
  original=align(foreground_probability(ORIGINAL/f'{cid}.npz'),ref,f'{cid}/original')
  seg_ref=nib.load(str(SEG_LABELS/f'{cid}_mask.nii.gz'))
  n120=align(foreground_probability(N120/f'{cid}.npz'),ref,f'{cid}/N120',nib.load(str(N120/f'{cid}.nii.gz')))
  s101=align(foreground_probability(S101/f'{cid}.npz'),ref,f'{cid}/S101',seg_ref)
  s50=align(foreground_probability(S50/f'{cid}.npz'),ref,f'{cid}/S50',seg_ref)
  components=[n120,s101,s50]
  tta=.6*components[0]+.2*components[1]+.2*components[2]
  save(OUT/'probabilities'/f'{cid}.npz',original);save(OUT/'tta_probabilities/axis0_flip'/f'{cid}.npz',tta)
  pred=(original>=THRESHOLD).astype(np.uint8);dest=OUT/'predictions'/f'{cid}.nii.gz';dest.parent.mkdir(parents=True,exist_ok=True);nib.save(nib.Nifti1Image(pred,ref.affine,ref.header),str(dest))
  print(f'[{i}/{len(cases)}] {cid}',flush=True)
 return 0
if __name__=='__main__':raise SystemExit(main())
