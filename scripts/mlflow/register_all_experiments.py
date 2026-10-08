#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,subprocess
from pathlib import Path
REPO=Path(__file__).resolve().parents[2];EXPORTS=REPO/'artifacts/mlflow_exports';URI='https://brainon-mlflow-3gykqo24da-as.a.run.app'
def clean(v,limit=500):
 if isinstance(v,(dict,list)):v=json.dumps(v,ensure_ascii=False)
 return str(v)[:limit]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--model-id',action='append');a=ap.parse_args();exports=[]
 for p in sorted(EXPORTS.glob('*.json')):
  x=json.loads(p.read_text())
  if a.model_id and x['experiment_id'] not in a.model_id:continue
  if x['status']!='COMPLETE':continue
  exports.append((p,x))
 if not a.execute:
  plans=[{'experiment_id':x['experiment_id'],'tracking':True,'registry':True,'registered_model_name':x['registered_model_name']} for _,x in exports];Path('artifacts/mlflow_registration_execution.json').write_text(json.dumps({'dry_run':True,'plans':plans},indent=2)+'\n');print(json.dumps({'dry_run':True,'tracking_runs':len(plans),'registry_versions':len(plans)},indent=2));return
 os.environ['MLFLOW_TRACKING_URI']=URI;os.environ['MLFLOW_TRACKING_TOKEN']=subprocess.check_output(['gcloud','auth','print-identity-token'],text=True).strip()
 import time,mlflow
 from mlflow import MlflowClient
 from mlflow.entities import Metric,Param,RunTag
 client=MlflowClient();results=[]
 for index,(p,x) in enumerate(exports,1):
  name=x['registered_model_name'];eid=x['experiment_id'];experiment=mlflow.set_experiment(name);existing=mlflow.search_runs([experiment.experiment_id],filter_string=f"tags.experiment_id = '{eid}'",max_results=1)
  if len(existing):run_id=existing.iloc[0].run_id;action='UPDATED_EXISTING_RUN'
  else:run_id=client.create_run(experiment.experiment_id,tags={'mlflow.runName':eid}).info.run_id;action='CREATED_RUN'
  tags={**x['tags'],'registry_eligible':'true'};metrics={**x['metrics']['original'],**{f'standard.{k}':v for k,v in x['metrics']['standardized'].items()}};now=int(time.time()*1000);metric_entities=[]
  for k,v in metrics.items():
   try:metric_entities.append(Metric(k,float(v),now,0))
   except Exception:pass
  client.log_batch(run_id,metrics=metric_entities,params=[Param(k,clean(v,250)) for k,v in x['params'].items()],tags=[RunTag(k,clean(v)) for k,v in tags.items()])
  client.log_artifact(run_id,str(p),artifact_path='inventory')
  for artifact in x['artifacts']:
   q=Path(artifact);q=q if q.is_absolute() else REPO/q
   if q.is_file():client.log_artifact(run_id,str(q),artifact_path='experiment_artifacts')
  client.set_terminated(run_id,status='FINISHED')
  try:client.get_registered_model(name)
  except Exception:client.create_registered_model(name,tags={'analysis_type':x['analysis_type']})
  versions=client.search_model_versions(f"name='{name}'");match=next((v for v in versions if v.tags.get('experiment_id')==eid),None)
  version=match.version if match else client.create_model_version(name=name,source=x['model_bundle']['gcs_uri'],run_id=run_id,await_creation_for=0,tags={**{k:clean(v) for k,v in x['tags'].items()},'experiment_id':eid,'lifecycle_status':'CANDIDATE'}).version
  results.append({'experiment_id':eid,'run_id':run_id,'action':action,'model_version':version});print(f'[{index}/{len(exports)}] {eid} run={run_id} version={version}',flush=True)
 Path('artifacts/mlflow_registration_execution.json').write_text(json.dumps({'dry_run':False,'results':results},indent=2)+'\n');print(json.dumps({'runs':len(results),'versions':len(results)},indent=2))
if __name__=='__main__':main()
