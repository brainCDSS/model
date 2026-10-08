#!/usr/bin/env python3
import argparse,json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import yaml
EXPERIMENTS=("H-T05","H-T06","H-T07","H-T08")
def now():return datetime.now(timezone.utc).isoformat()
def write_status(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+".tmp");tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2));os.replace(tmp,path)
def resolved_config(source,destination,resume):
    cfg=yaml.safe_load(source.read_text());cfg["experiment"]["continue_training"]=resume
    destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False));return destination
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--data-root",required=True,type=Path);parser.add_argument("--gpu",default="0");parser.add_argument("--resume",action="store_true");parser.add_argument("--dry-run",action="store_true");args=parser.parse_args()
    repository=Path(__file__).resolve().parents[2];results=repository/"runs/nnunet";batch=repository/"runs/batches/H-T05_H-T08"
    env=os.environ.copy();env.update({"CUDA_VISIBLE_DEVICES":str(args.gpu),"nnUNet_n_proc_DA":"0","nnUNet_def_n_proc":"1"});env["PYTHONPATH"]=os.pathsep.join([str(repository/"src"),env.get("PYTHONPATH","")]).rstrip(os.pathsep)
    prepare=[sys.executable,str(repository/"scripts/preprocessing/prepare_h_t05_h_t08_plans.py"),"--preprocessed-root",str(args.data_root.resolve()/"nnUNet_preprocessed")]
    if args.dry_run:print(" ".join(prepare))
    else:subprocess.run(prepare,check=True,env=env)
    commands={}
    for name in EXPERIMENTS:
        source=repository/f"configs/experiments/hemorrhage/tuning/{name}.yaml";config=resolved_config(source,batch/"resolved"/f"{name}.yaml",args.resume)
        commands[name]=[sys.executable,"-m","brainon_ai.nnunet.runner","--config",str(config),"--data-root",str(args.data_root.resolve()),"--results-root",str(results),"--no-validation"]
    validation_command = [
        sys.executable,
        str(repository / "scripts/validation/evaluate_h_t05_h_t08_neighborhood.py"),
        "--data-root",
        str(args.data_root.resolve()),
        "--radius",
        "4",
    ]
    if args.dry_run:
        for name,command in commands.items():print(name+": "+" ".join(command))
        print("Top-3 neighborhood validation: " + " ".join(validation_command))
        return 0
    status={"status":"running","started_at":now(),"current":None,"experiments":{}};write_status(batch/"batch_status.json",status)
    for name,command in commands.items():
        status["current"]=name;status["experiments"][name]={"status":"running","started_at":now()};write_status(batch/"batch_status.json",status)
        with (batch/f"{name}.log").open("a",encoding="utf-8") as log:code=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,text=True).wait()
        status["experiments"][name].update({"status":"completed" if code==0 else "failed","finished_at":now(),"exit_code":code});write_status(batch/"batch_status.json",status)
        if code:
            status.update({"status":"failed","current":None,"finished_at":now()});write_status(batch/"batch_status.json",status);return code
    status["current"] = "Top-3 neighborhood validation"
    status["phase"] = "full_validation"
    write_status(batch / "batch_status.json", status)
    with (batch / "top3_neighborhood_validation.log").open("a", encoding="utf-8") as log:
        code = subprocess.Popen(
            validation_command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        ).wait()
    if code:
        status.update({"status": "failed", "current": None, "finished_at": now(), "validation_exit_code": code})
        write_status(batch / "batch_status.json", status)
        return code
    status.update({"status":"completed","current":None,"phase":"completed","finished_at":now()});write_status(batch/"batch_status.json",status);return 0
if __name__=="__main__":raise SystemExit(main())
