#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import yaml
from brainon_ai.nnunet.runner import expected_output_folder,run_from_config

FOLDS=tuple(range(5))
def now():return datetime.now(timezone.utc).isoformat()
def write(path,payload):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+".tmp");tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2));os.replace(tmp,path)
def main():
 pa=argparse.ArgumentParser();pa.add_argument("--data-root",required=True,type=Path);pa.add_argument("--results-root",type=Path);pa.add_argument("--gpu",default="0");pa.add_argument("--folds",nargs="+",type=int,default=list(FOLDS));pa.add_argument("--dry-run",action="store_true");a=pa.parse_args()
 repo=Path(__file__).resolve().parents[3];results=(a.results_root or repo/"runs/nnunet").resolve();batch=repo/"runs/batches/H-T01-E300-KF";status_file=batch/"batch_status.json";folds=tuple(a.folds)
 if not folds or any(f not in FOLDS for f in folds):raise ValueError("--folds는 0 1 2 3 4 중 하나 이상이어야 합니다")
 env=os.environ.copy();env.update({"CUDA_VISIBLE_DEVICES":str(a.gpu),"nnUNet_n_proc_DA":"0","nnUNet_def_n_proc":"1"});env["PYTHONPATH"]=os.pathsep.join([str(repo/"src"),env.get("PYTHONPATH","")]).rstrip(os.pathsep)
 # run_from_config는 subprocess가 아니라 현재 Python 프로세스에서 Trainer를 생성하므로
 # 환경 사본뿐 아니라 os.environ에도 반드시 반영해야 한다.
 os.environ.update({key:env[key] for key in ("CUDA_VISIBLE_DEVICES","nnUNet_n_proc_DA","nnUNet_def_n_proc","PYTHONPATH")})
 prepare=[sys.executable,str(repo/"scripts/kfold/preprocessing/prepare_h_t01_e300_kfold_plans.py"),"--preprocessed-root",str(a.data_root.resolve()/"nnUNet_preprocessed")]
 print("PLAN:"," ".join(prepare),flush=True)
 if not a.dry_run:subprocess.run(prepare,check=True,env=env)
 status={"status":"dry_run" if a.dry_run else "running","started_at":now(),"current":None,"folds":{}}
 if not a.dry_run:write(status_file,status)
 for fold in folds:
  name=f"H-T01-E300-KF-F{fold}";config=repo/f"configs/experiments/hemorrhage/kfold/{name}.yaml";cfg=yaml.safe_load(config.read_text());output=expected_output_folder(cfg,results);latest=output/"checkpoint_latest.pth";final=output/"checkpoint_final.pth";best=output/"best_full_macro.json"
  if best.is_file():print(f"[skip complete] fold {fold}: {best}",flush=True);status["folds"][str(fold)]={"status":"completed","output":str(output)};continue
  mode="validate_only" if final.is_file() else ("resume" if latest.is_file() else "new")
  print(f"fold {fold}: mode={mode} output={output}",flush=True)
  validation=[sys.executable,str(repo/"scripts/kfold/validation/evaluate_h_t01_e300_kfold_neighborhood.py"),"--config",str(config),"--data-root",str(a.data_root.resolve()),"--results-root",str(results)]
  if a.dry_run:print("TRAIN:","skip" if mode=="validate_only" else str(config));print("VALIDATE:"," ".join(validation));continue
  status["current"]=name;status["folds"][str(fold)]={"status":"running","phase":"training","mode":mode,"started_at":now(),"output":str(output)};write(status_file,status)
  if mode!="validate_only":
   run_config=config
   if mode=="resume":
    cfg["experiment"]["continue_training"]=True;run_config=batch/f"resume_F{fold}.yaml";run_config.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False))
   try:run_from_config(run_config,a.data_root,results,run_validation=False)
   except BaseException as exc:
    status["status"]="failed";status["folds"][str(fold)].update({"status":"failed","error":repr(exc),"finished_at":now()});write(status_file,status);raise
  status["folds"][str(fold)]["phase"]="top3_neighborhood";write(status_file,status)
  log=batch/f"fold_{fold}_top3_neighborhood.log"
  with log.open("a",encoding="utf-8") as handle:code=subprocess.run(validation,env=env,stdout=handle,stderr=subprocess.STDOUT).returncode
  if code:
   status["status"]="failed";status["folds"][str(fold)].update({"status":"failed","exit_code":code,"finished_at":now()});write(status_file,status);return code
  status["folds"][str(fold)].update({"status":"completed","phase":"completed","finished_at":now()});write(status_file,status)
 status.update({"status":"completed","current":None,"finished_at":now()});write(status_file,status);return 0
if __name__=="__main__":raise SystemExit(main())
