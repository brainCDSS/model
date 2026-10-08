#!/usr/bin/env python3
"""Launch/resume A-B01, then refine its Top-3 full-volume neighborhoods."""
from __future__ import annotations
import argparse,json,os,subprocess
from pathlib import Path
import yaml

def main():
    pa=argparse.ArgumentParser(); pa.add_argument("--config",required=True,type=Path); pa.add_argument("--data-root",type=Path); pa.add_argument("--results-root",type=Path); pa.add_argument("--python-bin",type=Path); pa.add_argument("--dry-run",action="store_true"); pa.add_argument("--no-resume",action="store_true"); a=pa.parse_args()
    repo=Path(__file__).resolve().parents[3]; config=a.config.resolve(); c=yaml.safe_load(config.read_text(encoding="utf-8")); e,q,t=c["experiment"],c["plans"],c["trainer"]
    data=(a.data_root or Path(c["paths"]["data_root"])).resolve(); results=(a.results_root or Path(c["paths"]["results_root"])).resolve(); python=Path(os.path.abspath(a.python_bin or repo/".venv/bin/python")); train=python.parent/"nnUNetv2_train"; trainers=repo/"src/brainon_ai/models/aneurysm_binary/nnunet"
    dataset=f"Dataset{int(e['dataset_id']):03d}_{e['dataset_name']}"; pre=data/"nnUNet_preprocessed"; fold=results/dataset/f"nnUNetTrainerAneurysmBinaryBaseline__{q['output_identifier']}__{e['configuration']}"/f"fold_{e['fold']}"
    required={"Python":python,"nnUNetv2_train":train,"Trainer":trainers,"Dataset503":pre/dataset}; missing=[f"{n}: {x}" for n,x in required.items() if not x.exists()]
    if missing: raise FileNotFoundError("필수 경로가 없습니다:\n"+"\n".join(missing))
    resume=(fold/"checkpoint_latest.pth").is_file() and not a.no_resume
    prepare=[str(python),str(repo/"scripts/aneurysm_binary/preprocessing/prepare_a_b00_plans.py"),"--config",str(config),"--data-root",str(data)]
    command=[str(train),str(e["dataset_id"]),e["configuration"],str(e["fold"]),"-tr","nnUNetTrainerAneurysmBinaryBaseline","-p",q["output_identifier"],"-device",e.get("device","cuda")]+(["--c"] if resume else [])
    refine=[str(python),str(repo/"scripts/aneurysm_binary/validation/evaluate_a_b01_neighborhood.py"),"--config",str(config),"--data-root",str(data),"--results-root",str(results)]
    print(json.dumps({"experiment":"A-B01","mode":"resume" if resume else "new","output_folder":str(fold),"prepare":prepare,"train":command,"post_training_top3_neighborhood":refine},ensure_ascii=False,indent=2))
    if a.dry_run:return 0
    results.mkdir(parents=True,exist_ok=True); subprocess.run(prepare,check=True)
    env=os.environ.copy(); env.update({"nnUNet_raw":str(data/"nnUNet_raw"),"nnUNet_preprocessed":str(pre),"nnUNet_results":str(results),"nnUNet_extTrainer":str(trainers),"PYTHONPATH":os.pathsep.join([str(repo/"src"),str(trainers),env.get("PYTHONPATH","")]),"ICH_NNUNET_CONFIG":str(config),"nnUNet_compile":"false","nnUNet_n_proc_DA":"0"})
    code=subprocess.run(command,env=env,check=False).returncode
    if code:return code
    return subprocess.run(refine,env=env,check=False).returncode
if __name__=="__main__":raise SystemExit(main())
