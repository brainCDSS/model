#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,json,os,platform,re,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import yaml
REPO=Path(__file__).resolve().parents[2]
CONFIG_ROOT=REPO/'configs/experiments'; RUNS=REPO/'runs'; OUT=REPO/'artifacts'; EXPORTS=OUT/'mlflow_exports'; BUNDLES=OUT/'model_bundles'
MODEL_NAMES={'CT':'hemorrhagic-stroke-ct','MRI':'ischemic-stroke-mri','MRA':'cerebral-aneurysm-mra'}
ANALYSIS={'CT':'HEMORRHAGIC_STROKE_CT','MRI':'ISCHEMIC_STROKE_DWI_ADC','MRA':'CEREBRAL_ANEURYSM_MRA'}
MODALITY={'CT':'CT','MRI':'DWI_ADC','MRA':'MRA'}
SUMMARY_FILES={'CT':REPO/'reports/experiment_summaries/ct_hemorrhage_experiments_metrics.csv','MRI':REPO/'reports/experiment_summaries/mri_ischemia_experiments_metrics.csv','MRA':REPO/'reports/experiment_summaries/mra_aneurysm_experiments_metrics.csv'}
PATIENT_EXT=('.nii','.nii.gz','.npz','.pkl','.npy','.mha','.mhd','.dcm')
STANDARD_MAP={
'dice':['val_macro_dice_positive','val_macro_dice','validation_macro_dice','test_macro_dice_positive','val_standard_macro_dice_positive'],
'voxel_recall':['val_voxel_recall','validation_voxel_recall','test_voxel_recall_positive','val_standard_voxel_recall_positive'],
'voxel_precision':['val_voxel_precision','validation_voxel_precision','test_voxel_precision_positive','val_standard_voxel_precision_positive'],
'lesion_recall':['val_lesion_recall','test_lesion_recall','val_standard_lesion_recall'],
'lesion_precision':['val_lesion_precision','test_lesion_precision','val_standard_lesion_precision'],
'lesion_f1':['val_lesion_f1','test_lesion_f1','val_standard_lesion_f1'],
'case_detection_rate':['val_detection_rate','val_detection_rate_all_positive','test_case_detection_rate','val_standard_case_detection_rate'],
'negative_case_fp_rate':['val_negative_case_fp_rate','test_negative_case_fp_rate','val_standard_negative_case_fp_rate'],
'hd95':['val_hd95_mm','test_hd95_mm','val_standard_hd95_mm'],
'small_lesion_dice':['val_small_lesion_dice','test_small_lesion_dice','val_under5_dice'],
'small_lesion_recall':['val_small_lesion_recall','test_small_lesion_recall','val_under5_recall','val_detection_rate_under_5ml'],
'fp_per_case':['val_fp_per_scan','test_fp_per_case','val_standard_fp_per_case'],
'aneurysm_sensitivity':['val_lesion_recall','test_lesion_recall','val_standard_lesion_recall']}
def git(*args):
 r=subprocess.run(['git',*args],cwd=REPO,text=True,capture_output=True);return r.stdout.strip() if r.returncode==0 else 'MISSING'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def load_yaml(p):
 try:return yaml.safe_load(p.read_text()) or {}
 except Exception:return {}
def exp_id(p,cfg):
 if 'smoke' in p.stem.lower(): return p.stem
 value=cfg.get('experiment_id') or cfg.get('experiment',{}).get('id')
 if value:return str(value)
 return p.stem
def disease(eid,path):
 if eid.startswith('H-') or 'hemorrhage' in str(path):return 'CT'
 if eid.startswith('I-') or 'ischemia' in str(path):return 'MRI'
 if eid.startswith('A-') or 'aneurysm' in str(path):return 'MRA'
 return 'UNKNOWN'
def flatten_params(cfg):
 def g(*paths):
  for path in paths:
   cur=cfg
   for key in path.split('.'):
    if not isinstance(cur,dict) or key not in cur:cur=None;break
    cur=cur[key]
   if cur is not None:return cur
  return 'MISSING'
 params={'loss':g('loss.name'),'optimizer':g('trainer.optimizer.name','training.optimizer.name','optimizer.name'),'learning_rate':g('trainer.learning_rate','training.learning_rate'),'scheduler':g('trainer.scheduler.name','training.scheduler.name','scheduler.name'),'weight_decay':g('trainer.weight_decay','training.weight_decay'),'epochs':g('trainer.epochs','training.epochs'),'selected_epoch':'MISSING','batch_size':g('plans.batch_size','data.batch_size','training.batch_size'),'patch_size':g('plans.patch_size','data.patch_size'),'spacing':g('preprocessing.spacing'),'threshold':g('evaluation.threshold','probability_evaluation.selected_threshold'),'sampling_strategy':g('sampling.strategy','sampling.mode'),'oversample_foreground_percent':g('trainer.oversample_foreground_percent'),'deep_supervision':g('model.deep_supervision_depth','network.deep_supervision'),'augmentation':g('augmentation'),'fold':g('experiment.fold','fold')}
 for section in ('network','model','plans','sampling','ensemble','loss'):
  if isinstance(cfg.get(section),dict):
   for k,v in cfg[section].items():
    if isinstance(v,(str,int,float,bool,list)) and k not in params:params[f'{section}.{k}']=v
 return params
def summary_rows():
 out={}
 for kind,p in SUMMARY_FILES.items():
  if not p.is_file():continue
  with p.open(encoding='utf-8-sig',newline='') as f:
   for row in csv.DictReader(f):
    eid=row.get('experiment_id','').strip()
    if eid:out[(kind,eid)]=row
 return out
def numeric_metrics(row):
 original={}
 for k,v in (row or {}).items():
  if not v or k in ('experiment_id','disease','method','model_family','config_path','test_metrics_path','metric_source','val_checkpoint','val_summary','val_experiment'):continue
  try: original[k]=float(v)
  except Exception:pass
 standard={}
 for target,sources in STANDARD_MAP.items():
  for source in sources:
   if source in original:standard[target]=original[source];break
 return original,standard
def find_model_dir(cfg,eid,kind):
 candidates=[]
 if kind=='MRI' and eid.startswith('I-S'):candidates += [RUNS/'I-segresnet'/eid,RUNS/'ischemia'/eid]
 if kind=='CT' and eid.startswith('H-S'):candidates += [RUNS/'segresnet'/eid,RUNS/'hemorrhage'/eid]
 if kind=='MRA' and eid.startswith('A-S'):candidates += [RUNS/'aneurysm'/eid]
 if eid=='H-FM10':candidates += [RUNS/eid]
 plans=cfg.get('plans',{}).get('output_identifier');conf=cfg.get('experiment',{}).get('configuration','3d_fullres');fold=cfg.get('experiment',{}).get('fold',0)
 if plans:
  pattern=f'*__{plans}__{conf}'
  for d in RUNS.rglob(pattern):
   if d.is_dir():candidates.insert(0,d/f'fold_{fold}')
 candidates += [RUNS/eid,RUNS/'aneurysm_binary'/eid]
 return next((x for x in candidates if x.exists()),None)
def checkpoints(folder,row):
 found=[]
 explicit=(row or {}).get('val_checkpoint')
 if explicit and Path(explicit).is_file():found.append(Path(explicit))
 if folder:
  for name in ('checkpoint_best_dual_criteria.pth','checkpoint_best_composite.pth','checkpoint_best_full_macro.pth','checkpoint_best.pth','checkpoint_final.pth','model.pt','best_model.pth'):
   q=folder/name
   if q.is_file():found.append(q)
  q=folder/'checkpoints'
  if q.is_dir():
   for name in ('checkpoint_best_dual_criteria.pth','checkpoint_best_composite.pth','checkpoint_best_full_macro.pth','checkpoint_best.pth'):
    x=q/name
    if x.is_file():found.append(x)
 unique=[]
 for x in found:
  if x not in unique:unique.append(x)
 return unique
def evaluation_files(folder,row,eid):
 files=[]
 for key in ('test_metrics_path','val_summary','val_standard_metrics_path','metric_source'):
  v=(row or {}).get(key)
  if v:
   q=Path(v);q=q if q.is_absolute() else (REPO/q if not str(q).startswith('model/') else REPO.parent/q)
   if q.is_file():files.append(q)
 for base in filter(None,[folder,RUNS/'final_evaluations'/eid,RUNS/'final_evaluations'/f'{eid}-VAL']):
  for rel in ('evaluation_metrics.json','custom_metrics.json','final_selection.json','final_metrics/summary.json','final_metrics/summary_metrics.csv','progress.json','training_history.csv','history.csv'):
   q=base/rel
   if q.is_file():files.append(q)
 if folder and (folder/'full_validation').is_dir():
  summaries=sorted((folder/'full_validation').glob('epoch_*/summary.json'))
  if summaries:files.append(summaries[-1])
 return list(dict.fromkeys(files))
def safe_artifacts(config,folder,evaluations):
 files=[config,*evaluations]
 if folder:
  for pattern in ('*.log','training_log*.txt','progress.png','history.csv','training_history.csv','debug.json'):
   files += sorted(folder.glob(pattern))
 clean=[]
 for p in files:
  if p.is_file() and not p.name.endswith(PATIENT_EXT) and not any(x in p.parts for x in ('predictions','probabilities','imagesTr','imagesTs','labelsTr','labelsTs')):clean.append(p)
 return list(dict.fromkeys(clean))
def bundle_files(config,folder,checkpoint,kind,cfg):
 files=[config]
 if checkpoint:files.append(checkpoint)
 if folder:
  model_dir=folder.parent if folder.name.startswith('fold_') else folder
  for base in (folder,model_dir):
   for name in ('plans.json','dataset.json','inference_config.yaml','inference_config.json','postprocessing.json','postprocessing.pkl'):
    q=base/name
    if q.is_file():files.append(q)
 return list(dict.fromkeys(files))
def environment():
 versions={'python_version':platform.python_version()}
 for module,key in [('torch','torch_version'),('monai','monai_version'),('nnunetv2','nnunet_version')]:
  try:
   m=__import__(module);versions[key]=getattr(m,'__version__','installed_version_unknown')
  except Exception:versions[key]='MISSING'
 try:
  import torch;versions['cuda_version']=torch.version.cuda or 'MISSING'
 except Exception:versions['cuda_version']='MISSING'
 return versions
def discover_ensembles(records):
 for p in sorted((RUNS/'ischemia_ensemble').glob('**/ensemble_summary.json')):
  rel=p.relative_to(RUNS/'ischemia_ensemble');parts=rel.parts
  if parts[0].startswith('I-E01'):eid='I-E01'
  elif parts[0].startswith('I-E02'):eid='I-E02-'+parts[1]
  elif parts[0].startswith('I-E03'):eid='I-E03-'+parts[1]
  else:eid=parts[0].split('_')[0]
  records.append({'experiment_id':eid,'config_path':None,'config':{'ensemble':{}},'kind':'MRI','ensemble_summary':p})
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--hash-checkpoints',action='store_true');a=ap.parse_args();OUT.mkdir(exist_ok=True);EXPORTS.mkdir(parents=True,exist_ok=True);BUNDLES.mkdir(parents=True,exist_ok=True)
 rows=summary_rows();records=[]
 for p in sorted(CONFIG_ROOT.rglob('*')):
  if p.suffix not in ('.yaml','.yml','.json') or p.name=='matrix.yaml':continue
  cfg=load_yaml(p);eid=exp_id(p,cfg)
  if not re.match(r'^[HIA]-',eid):continue
  records.append({'experiment_id':eid,'config_path':p,'config':cfg,'kind':disease(eid,p)})
 discover_ensembles(records)
 dedup={}
 for r in records:
  key=r['experiment_id'];dedup[key]=r if key not in dedup or r.get('config_path') else dedup[key]
 records=list(dedup.values());env=environment();branch=git('branch','--show-current');commit=git('rev-parse','HEAD');remote=git('config','--get','remote.origin.url');inventory=[];plans=[];uploads=[];missing_lines=['# MLflow missing report','','## Global blockers','','- MRI requested prefix `ischemic-stroke-mri` conflicts with existing GCS prefix `ischemic-stroke-dwi-adc`; MRI uploads remain dry-run until naming is approved.','']
 failed_ids={'I-S60','I-S100'}
 for r in sorted(records,key=lambda x:x['experiment_id']):
  eid=r['experiment_id'];kind=r['kind'];cfg=r['config'];config=r.get('config_path');row=rows.get((kind,eid),{});folder=find_model_dir(cfg,eid,kind) if config else None
  cks=checkpoints(folder,row);checkpoint=cks[0] if cks else None;evals=evaluation_files(folder,row,eid)
  if r.get('ensemble_summary'):evals.append(r['ensemble_summary'])
  ensemble_cfg=cfg.get('ensemble',{}) if isinstance(cfg.get('ensemble'),dict) else {}
  ensemble_summary=ensemble_cfg.get('summary')
  if ensemble_summary:
   q=Path(ensemble_summary);q=q if q.is_absolute() else REPO/q
   if q.is_file():evals.append(q)
  original,standard=numeric_metrics(row)
  summary_for_metrics=r.get('ensemble_summary') or (Path(ensemble_summary) if ensemble_summary and Path(ensemble_summary).is_absolute() else REPO/ensemble_summary if ensemble_summary else None)
  if summary_for_metrics and Path(summary_for_metrics).is_file():
   try:
    data=json.loads(Path(summary_for_metrics).read_text());data=data.get('selected',data)
    for k,v in data.items():
     if isinstance(v,(int,float)):original[k]=float(v)
   except Exception:pass
  has_eval=bool(evals or original)
  smoke='smoke' in eid.lower() or (config and 'smoke' in config.name.lower())
  run_evidence=folder is not None or bool(evals)
  component_paths=[]
  for value in ensemble_cfg.get('components',{}).values():
   q=Path(value);q=q if q.is_absolute() else REPO/q
   if q.is_file():component_paths.append(q)
  components_complete=bool(ensemble_cfg.get('components')) and len(component_paths)==len(ensemble_cfg.get('components',{}))
  if smoke:status='SMOKE'
  elif eid in failed_ids:status='FAILED'
  elif ((checkpoint is not None) or components_complete) and has_eval:status='COMPLETE'
  elif run_evidence or checkpoint or has_eval:status='PARTIAL'
  else:status='UNKNOWN'
  framework='Ensemble' if ensemble_cfg or re.match(r'^[HIA]-E',eid) else 'SegResNet' if '-S' in eid else 'NV-Segment-CT' if eid.startswith('H-FM') else 'nnU-Net'
  architecture=cfg.get('network',{}).get('name') or cfg.get('model',{}).get('name') or ('soft_voting_ensemble' if framework=='Ensemble' else framework)
  trainer_class='MISSING'
  if folder:
   m=re.search(r'(nnUNetTrainer[^_]+(?:_[^_]+)?)__',str(folder));trainer_class=m.group(1) if m else ('SegResNet' if framework=='SegResNet' else 'MISSING')
  params=flatten_params(cfg);selected=(row or {}).get('val_epoch');params['selected_epoch']=float(selected) if selected and str(selected).replace('.','',1).isdigit() else params['selected_epoch']
  model_name=MODEL_NAMES[kind];gcs=f'gs://brainon-cdss-models/{model_name}/{eid}'
  bundle=bundle_files(config,folder,checkpoint,kind,cfg) if config else ([r['ensemble_summary']] if r.get('ensemble_summary') else [])
  if framework=='Ensemble': bundle=list(dict.fromkeys(([config] if config else [])+([Path(summary_for_metrics)] if summary_for_metrics else [])+component_paths))
  artifacts=safe_artifacts(config,folder,evals) if config else evals
  missing=[]
  if not checkpoint and framework!='Ensemble':missing.append('checkpoint')
  if framework=='Ensemble' and not components_complete:missing.append('component_checkpoint_binding')
  if not has_eval:missing.append('evaluation')
  if not config and framework!='Ensemble':missing.append('experiment_config')
  required_names={x.name for x in bundle}
  bundle_ready=(bool(checkpoint and config) and (framework!='nnU-Net' or {'plans.json','dataset.json'}.issubset(required_names))) if framework!='Ensemble' else bool(config and components_complete and summary_for_metrics)
  registry_eligible=status=='COMPLETE' and bundle_ready
  if checkpoint: cp_hash=sha(checkpoint) if a.hash_checkpoints else 'PENDING_SHA256'
  elif component_paths: cp_hash=hashlib.sha256(''.join(sha(x) for x in component_paths).encode()).hexdigest() if a.hash_checkpoints else 'PENDING_SHA256'
  else: cp_hash='MISSING'
  tags={'experiment_id':eid,'model_name':model_name,'analysis_type':ANALYSIS[kind],'modality':MODALITY[kind],'framework':framework,'architecture':architecture,'trainer_class':trainer_class,'dataset':cfg.get('experiment',{}).get('dataset_name',cfg.get('data',{}).get('dataset','MISSING')),'input_channels':cfg.get('data',{}).get('in_channels','MISSING'),'channel_order':cfg.get('data',{}).get('modalities','MISSING'),'source_repository':remote,'source_branch':branch,'source_git_commit':commit,'source_config':str(config.relative_to(REPO)) if config else 'MISSING','checkpoint_name':checkpoint.name if checkpoint else ','.join(x.name for x in component_paths) if component_paths else 'MISSING','checkpoint_sha256':cp_hash,**env,'model_bundle_uri':gcs,'manifest_uri':gcs+'/manifest.json','checkpoint_uri':gcs+'/'+checkpoint.name if checkpoint else gcs+'/components/' if component_paths else 'MISSING','experiment_status':status}
  export={'experiment_id':eid,'status':status,'registry_eligible':registry_eligible,'registered_model_name':model_name,'analysis_type':ANALYSIS[kind],'modality':MODALITY[kind],'framework':framework,'architecture':architecture,'metrics':{'original':original,'standardized':standard},'params':params,'tags':tags,'artifacts':[str(x.relative_to(REPO)) if x.is_relative_to(REPO) else str(x) for x in artifacts],'model_bundle':{'files':[{'source':str(x),'name':x.name,'sha256':cp_hash if x==checkpoint else 'PENDING_SHA256'} for x in bundle],'gcs_uri':gcs},'missing':missing}
  filename=re.sub(r'[^A-Za-z0-9_.-]+','_',eid)+'.json';(EXPORTS/filename).write_text(json.dumps(export,ensure_ascii=False,indent=2,default=str)+'\n')
  manifest={'schema_version':'1.0','experiment_id':eid,'registered_model_name':model_name,'source_git_commit':commit,'files':export['model_bundle']['files'],'gcs_uri':gcs,'patient_data_included':False};bd=BUNDLES/re.sub(r'[^A-Za-z0-9_.-]+','_',eid);bd.mkdir(parents=True,exist_ok=True);(bd/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
  item={'experiment_id':eid,'analysis_type':ANALYSIS[kind],'architecture':architecture,'framework':framework,'checkpoint':str(checkpoint) if checkpoint else 'MISSING','evaluation':[str(x) for x in evals] or ['MISSING'],'metric_count':len(original)+len(standard),'param_count':sum(v!='MISSING' for v in params.values()),'training_history':next((str(x) for x in artifacts if 'history' in x.name or 'log' in x.name),'MISSING'),'bundle_ready':bundle_ready,'mlflow_ready':status in ('COMPLETE','PARTIAL','SMOKE'),'registry_eligible':registry_eligible,'missing':missing,'status':status,'export_manifest':str(EXPORTS/filename)};inventory.append(item)
  plans.append({'experiment_id':eid,'action':'TRACKING_AND_REGISTRY_CANDIDATE' if registry_eligible else 'INVENTORY_ONLY','registered_model_name':model_name,'export_manifest':str(EXPORTS/filename),'status':status})
  uploads.append({'experiment_id':eid,'enabled':registry_eligible and kind!='MRI','source_files':[str(x) for x in bundle],'destination':gcs,'conflict_policy':'same_sha256_skip_different_sha256_conflict','patient_data_allowed':False,'blocker':'MRI_PREFIX_CONFLICT' if kind=='MRI' else None})
  if missing:missing_lines += [f'## {eid} ({status})','',*['- '+x for x in missing],'']
 counts={s:sum(x['status']==s for x in inventory) for s in ('COMPLETE','PARTIAL','FAILED','SMOKE','UNKNOWN')}
 (OUT/'all_experiments_inventory.json').write_text(json.dumps({'generated_at':datetime.now(timezone.utc).isoformat(),'source_git_commit':commit,'counts':counts,'experiments':inventory},ensure_ascii=False,indent=2)+'\n')
 schema={'schema_version':'1.0','mapping_policy':'preserve original metric and add standardized metric only when semantics match','standard_metric_mapping':STANDARD_MAP,'common_metrics':['dice','voxel_recall','voxel_precision','lesion_recall','lesion_precision','lesion_f1','case_detection_rate','negative_case_fp_rate','hd95'],'ct_specific':['small_lesion_dice','small_lesion_recall','dice_lt_5ml','recall_lt_5ml','hemorrhage_volume_mae_ml','hemorrhage_volume_error_percent','fp_volume_ml','fn_volume_ml'],'mri_specific':['small_lesion_dice','small_lesion_recall','dice_small','dice_medium','dice_large','lesion_volume_mae_ml','lesion_volume_error_percent'],'mra_specific':['aneurysm_sensitivity','small_aneurysm_recall','fp_per_case']};(OUT/'mlflow_metric_schema.json').write_text(json.dumps(schema,indent=2)+'\n')
 (OUT/'mlflow_registration_plan.json').write_text(json.dumps({'dry_run':True,'tracking_uri':'https://brainon-mlflow-3gykqo24da-as.a.run.app','registered_models':MODEL_NAMES,'runs':plans},indent=2)+'\n')
 (OUT/'gcs_upload_plan.json').write_text(json.dumps({'dry_run':True,'bucket':'gs://brainon-cdss-models','uploads':uploads},indent=2)+'\n')
 (OUT/'mlflow_missing_report.md').write_text('\n'.join(missing_lines)+'\n')
 print(json.dumps({'experiments':len(inventory),'counts':counts,'registry_eligible':sum(x['registry_eligible'] for x in inventory),'exports':len(list(EXPORTS.glob('*.json')))},indent=2))
if __name__=='__main__':main()
