#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import yaml
EXPERIMENTS=("H-N80","H-N90")
def now():return datetime.now(timezone.utc).isoformat()
def write_status(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(payload,ensure_ascii=False,indent=2));os.replace(temporary,path)
def resolved_config(source,destination,resume):
    config=yaml.safe_load(source.read_text());config["experiment"]["continue_training"]=resume
    destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False));return destination
def main():
    parser=argparse.ArgumentParser(description="H-N80 다음 H-N90 순차 실행")
    parser.add_argument("--data-root",type=Path,required=True);parser.add_argument("--gpu",default="0")
    parser.add_argument("--resume",action="store_true");parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--no-validation",action="store_true");args=parser.parse_args()
    repository=Path(__file__).resolve().parents[2];preprocessed=args.data_root.resolve()/"nnUNet_preprocessed"
    results=repository/"runs/nnunet";batch_root=repository/"runs/batches/H-N80_H-N90"
    env=os.environ.copy();env.update({"CUDA_VISIBLE_DEVICES":str(args.gpu),"nnUNet_n_proc_DA":"0","nnUNet_def_n_proc":"1"})
    env["PYTHONPATH"]=os.pathsep.join([str(repository/"src"),env.get("PYTHONPATH","")]).rstrip(os.pathsep)
    prepare=[sys.executable,str(repository/"scripts/preprocessing/prepare_h_n80_h_n90_plans.py"),"--preprocessed-root",str(preprocessed)]
    if args.dry_run:print(" ".join(prepare))
    else:subprocess.run(prepare,check=True,env=env)
    commands={}
    for name in EXPERIMENTS:
        source=repository/"configs/experiments/hemorrhage"/f"{name}.yaml"
        config=resolved_config(source,batch_root/"resolved"/f"{name}.yaml",args.resume)
        command=[sys.executable,"-m","brainon_ai.nnunet.runner","--config",str(config),"--data-root",str(args.data_root.resolve()),"--results-root",str(results)]
        if args.no_validation:command.append("--no-validation")
        commands[name]=command
    if args.dry_run:
        for name,command in commands.items():print(name+": "+" ".join(command))
        return 0
    status={"status":"running","started_at":now(),"current":None,"experiments":{}}
    write_status(batch_root/"batch_status.json",status)
    for name,command in commands.items():
        status["current"]=name;status["experiments"][name]={"status":"running","started_at":now()};write_status(batch_root/"batch_status.json",status)
        with (batch_root/f"{name}.log").open("a",encoding="utf-8") as log:
            code=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,text=True).wait()
        status["experiments"][name].update({"status":"completed" if code==0 else "failed","finished_at":now(),"exit_code":code});write_status(batch_root/"batch_status.json",status)
        if code:
            status.update({"status":"failed","current":None,"finished_at":now()});write_status(batch_root/"batch_status.json",status);return code
    status.update({"status":"completed","current":None,"finished_at":now()});write_status(batch_root/"batch_status.json",status);return 0
if __name__=="__main__":raise SystemExit(main())
