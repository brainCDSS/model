#!/usr/bin/env python3
import csv,json
from pathlib import Path
REPO=Path(__file__).resolve().parents[2];ART=REPO/'artifacts';bad=[];rows=[]
for p in sorted((ART/'mlflow_exports').glob('*.json')):
 x=json.loads(p.read_text());required=['experiment_id','status','registry_eligible','registered_model_name','metrics','params','tags','artifacts','model_bundle']
 missing=[k for k in required if k not in x]
 if missing:bad.append(f'{p.name}: missing keys {missing}')
 for a in x.get('artifacts',[]):
  low=a.lower()
  if any(z in low for z in ('/prediction','/probabilit','/imagestr','/imagests','/labelstr','/labelsts')) or low.endswith(('.nii','.nii.gz','.npz','.npy','.dcm')):bad.append(f'{p.name}: patient-like artifact {a}')
 for f in x.get('model_bundle',{}).get('files',[]):
  if not Path(f['source']).is_file():bad.append(f"{p.name}: missing bundle source {f['source']}")
 rows.append({'Experiment':x['experiment_id'],'Analysis Type':x['analysis_type'],'Architecture':x['architecture'],'Framework':x['framework'],'Checkpoint':x['tags']['checkpoint_name'],'Evaluation':'YES' if x['metrics']['original'] else 'FILE_ONLY' if x['artifacts'] else 'NO','Metrics':len(x['metrics']['original'])+len(x['metrics']['standardized']),'Params':sum(v!='MISSING' for v in x['params'].values()),'Training History':'YES' if any('history' in a or 'log' in a for a in x['artifacts']) else 'NO','Bundle Ready':'YES' if x['registry_eligible'] else 'NO','MLflow Ready':'YES' if x['status'] in ('COMPLETE','PARTIAL','SMOKE') else 'NO','Registry Eligible':'YES' if x['registry_eligible'] else 'NO','Missing':';'.join(x.get('missing',[])),'Status':x['status']})
with (ART/'all_experiments_inventory.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
report={'valid':not bad,'experiment_count':len(rows),'errors':bad};(ART/'inventory_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if bad:raise SystemExit(1)
