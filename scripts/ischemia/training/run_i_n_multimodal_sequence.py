#!/usr/bin/env python3
"""Run I-N110 then I-N120 sequentially; each experiment resumes safely."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

DEFAULTS = ("I-N110", "I-N120")

def completed(repo: Path, config: Path) -> bool:
    import yaml
    c=yaml.safe_load(config.read_text(encoding="utf-8")); e,p=c["experiment"],c["plans"]
    fold=Path(c["paths"]["results_root"])/f"Dataset{int(e['dataset_id']):03d}_{e['dataset_name']}"/f"nnUNetTrainerIschemiaMultimodal__{p['output_identifier']}__{e['configuration']}"/f"fold_{e['fold']}"
    return (fold/"final_selection.json").is_file() and (fold/"best_probability_validation").is_dir()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dry-run",action="store_true"); ap.add_argument("--rerun-completed",action="store_true"); a=ap.parse_args()
    repo=Path(__file__).resolve().parents[3]; py=repo/".venv/bin/python"; launcher=repo/"scripts/ischemia/training/run_i_n_multimodal.py"
    status=[]
    for eid in DEFAULTS:
        cfg=repo/f"configs/experiments/ischemia/{eid}.yaml"
        if completed(repo,cfg) and not a.rerun_completed and not a.dry_run:
            print(f"[SKIP] {eid}: final selection/probability maps already exist",flush=True); status.append({"experiment":eid,"status":"skipped_completed"}); continue
        cmd=[str(py),str(launcher),"--config",str(cfg)]+(["--dry-run"] if a.dry_run else [])
        print(f"[START] {eid}",flush=True); code=subprocess.run(cmd,cwd=repo,check=False).returncode
        status.append({"experiment":eid,"returncode":code});
        if code: print(json.dumps(status,ensure_ascii=False,indent=2)); return code
        print(f"[DONE] {eid}",flush=True)
    print(json.dumps(status,ensure_ascii=False,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
