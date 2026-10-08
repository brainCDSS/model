#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,subprocess
from pathlib import Path
REPO=Path(__file__).resolve().parents[2];EXPORTS=REPO/'artifacts/mlflow_exports';BUNDLES=REPO/'artifacts/model_bundles'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--model-id',action='append');a=ap.parse_args();actions=[]
 for ep in sorted(EXPORTS.glob('*.json')):
  x=json.loads(ep.read_text());eid=x['experiment_id']
  if not x['registry_eligible'] or (a.model_id and eid not in a.model_id):continue
  destination=x['model_bundle']['gcs_uri'].rstrip('/');files=[Path(f['source']) for f in x['model_bundle']['files']];manifest=BUNDLES/ep.stem/'manifest.json';files.append(manifest)
  names=[x.name for x in files]
  for index,source in enumerate(files):
   if source==manifest: relative='manifest.json'
   elif names.count(source.name)>1: relative=f'components/{index:02d}_{source.name}'
   else: relative=source.name
   target=destination+'/'+relative;sha=digest(source);action={'experiment_id':eid,'source':str(source),'target':target,'sha256':sha,'action':'UPLOAD'}
   if a.execute:
    desc=subprocess.run(['gcloud','storage','objects','describe',target,'--format=value(metadata.sha256)'],text=True,capture_output=True)
    if desc.returncode==0:
     remote=desc.stdout.strip()
     if remote==sha:action['action']='SKIP_SAME_SHA256'
     else:action['action']='CONFLICT';actions.append(action);continue
    if action['action']=='UPLOAD':subprocess.run(['gcloud','storage','cp','--custom-metadata',f'sha256={sha}',str(source),target],check=True)
   actions.append(action)
 out=REPO/'artifacts/gcs_upload_execution.json';out.write_text(json.dumps({'dry_run':not a.execute,'actions':actions},indent=2)+'\n');print(json.dumps({'dry_run':not a.execute,'files':len(actions),'upload':sum(x['action']=='UPLOAD' for x in actions),'skip':sum(x['action'].startswith('SKIP') for x in actions),'conflict':sum(x['action']=='CONFLICT' for x in actions)},indent=2))
 if any(x['action']=='CONFLICT' for x in actions):raise SystemExit(2)
if __name__=='__main__':main()
