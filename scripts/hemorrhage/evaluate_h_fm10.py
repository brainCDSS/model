#!/usr/bin/env python3
"""Export H-FM10 probabilities in original CT space and evaluate thresholds."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.inferers import SlidingWindowInferer
from monai.transforms import CastToTyped, Compose, CropForegroundd, EnsureTyped, Invertd, LoadImaged, Orientationd, ScaleIntensityRangePercentilesd, SpatialPadd, Spacingd
from scipy.ndimage import label

STRUCTURE=np.ones((3,3,3),np.uint8)
def cc(mask): return label(mask,structure=STRUCTURE)
def build_transforms(patch,spacing):
    return Compose([
        LoadImaged(keys='image',image_only=True,ensure_channel_first=True),
        CropForegroundd(keys='image',source_key='image',margin=10,allow_smaller=True,start_coord_key=None,end_coord_key=None),
        ScaleIntensityRangePercentilesd(keys='image',lower=1,upper=99,b_min=0,b_max=1,clip=True),
        Orientationd(keys='image',axcodes='RAS'),
        Spacingd(keys='image',pixdim=spacing,mode='bilinear'),
        CastToTyped(keys='image',dtype=torch.float32), EnsureTyped(keys='image',track_meta=True),
        SpatialPadd(keys='image',spatial_size=patch,mode='constant'),
    ])
def lesion_metrics(prob,gt,threshold,voxel_ml):
    pred=prob>=threshold; gc,gn=cc(gt); pc,pn=cc(pred)
    detected=sum(bool(pred[gc==i].any()) for i in range(1,gn+1)); fp=sum(not bool(gt[pc==i].any()) for i in range(1,pn+1))
    tp=np.logical_and(pred,gt).sum(); f=np.logical_and(pred,~gt).sum(); fn=np.logical_and(~pred,gt).sum()
    den=2*tp+f+fn
    return dict(dice=2*tp/den if den else 1., voxel_recall=tp/(tp+fn) if tp+fn else 1., voxel_precision=tp/(tp+f) if tp+f else 1., detected=detected, gt_lesions=gn, pred_lesions=pn, fp_lesions=fp)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--bundle-dir',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--datalist',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--limit',type=int);ap.add_argument('--patch-size',type=int,nargs=3,default=[128,128,96]);ap.add_argument('--spacing',type=float,nargs=3,default=[1.5,1.5,1.5]);ap.add_argument('--device',default='cuda');a=ap.parse_args()
 sys.path.insert(0,str(a.bundle_dir)); from scripts.vista3d_softmax import Vista3dSoftmax
 records=[r for r in json.loads(a.datalist.read_text())['training'] if int(r['fold'])==0]
 if a.limit: records=records[:a.limit]
 device=torch.device(a.device if a.device=='cpu' or torch.cuda.is_available() else 'cpu')
 model=Vista3dSoftmax(in_channels=1,class_ids=[22]).to(device); model.load_state_dict(torch.load(a.checkpoint,map_location=device),strict=True);model.eval()
 pre=build_transforms(a.patch_size,a.spacing); inverse=Invertd(keys='probability',transform=pre,orig_keys='image',nearest_interp=False,to_tensor=True)
 infer=SlidingWindowInferer(roi_size=a.patch_size,sw_batch_size=1,overlap=.625,mode='gaussian')
 prob_dir=a.output_dir/'probabilities';prob_dir.mkdir(parents=True,exist_ok=True)
 rows=[]
 with torch.no_grad():
  for index,r in enumerate(records,1):
   case_id=r['case_id']; data=pre({'image':r['image']}); image=data['image'].unsqueeze(0).to(device)
   with torch.autocast(device_type='cuda',enabled=device.type=='cuda'):
    probability=torch.softmax(infer(image,model),dim=1)[0].cpu()
   restored=inverse({'image':data['image'],'probability':probability})['probability']
   probability=np.asarray(restored,dtype=np.float32)
   reference=nib.load(r['label']); gt=np.asanyarray(reference.dataobj)>0
   if probability.shape[1:] != gt.shape: raise RuntimeError(f'{case_id}: probability {probability.shape} != GT {gt.shape}')
   np.savez_compressed(prob_dir/f'{case_id}.npz',probabilities=probability)
   voxel_ml=float(np.prod(reference.header.get_zooms()[:3]))/1000
   row={'case_id':case_id}
   for threshold in [.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9]:
    mask=probability[1]>=threshold
    nib.save(nib.Nifti1Image(mask.astype(np.uint8),reference.affine,reference.header),prob_dir/f'{case_id}_t{threshold:.2f}.nii.gz')
   rows.append(row);print(f'[{index}/{len(records)}] {case_id}',flush=True)
 # evaluate all stored probabilities once in original NIfTI space
 threshold_rows=[]
 for threshold in [.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9]:
  values=[]
  for r in records:
   case_id=r['case_id'];reference=nib.load(r['label']);gt=np.asanyarray(reference.dataobj)>0;prob=np.load(prob_dir/f'{case_id}.npz')['probabilities'][1]
   values.append(lesion_metrics(prob,gt,threshold,float(np.prod(reference.header.get_zooms()[:3]))/1000))
  total_gt=sum(x['gt_lesions'] for x in values);total_detected=sum(x['detected'] for x in values);total_pred=sum(x['pred_lesions'] for x in values);total_fp=sum(x['fp_lesions'] for x in values);prec=(total_pred-total_fp)/total_pred if total_pred else 1.;rec=total_detected/total_gt if total_gt else 1.;f2=5*prec*rec/(4*prec+rec) if 4*prec+rec else 0.
  threshold_rows.append({'threshold':threshold,'macro_dice':float(np.mean([x['dice'] for x in values])),'voxel_recall':float(np.mean([x['voxel_recall'] for x in values])),'voxel_precision':float(np.mean([x['voxel_precision'] for x in values])),'lesion_recall':rec,'lesion_precision':prec,'lesion_f2':f2,'fp_per_scan':total_fp/len(values),'detected_lesions':total_detected,'gt_lesions':total_gt})
 df=pd.DataFrame(threshold_rows);df.to_csv(a.output_dir/'threshold_sweep.csv',index=False);eligible=df[df.fp_per_scan<=1.5];best=(eligible if len(eligible) else df).sort_values(['lesion_f2','macro_dice'],ascending=False).iloc[0].to_dict();(a.output_dir/'summary.json').write_text(json.dumps({'cases':len(records),'checkpoint':str(a.checkpoint),'recommended_lesion_f2':best},ensure_ascii=False,indent=2)+'\n');print(json.dumps(best,ensure_ascii=False,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
