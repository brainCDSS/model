#!/usr/bin/env python3
"""Run the seven hy SegResNet experiments sequentially with automatic resume."""
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path
DEFAULTS=("I-S40","I-S50","I-S60","I-S70","I-S80","I-S90","I-S100")
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dry-run",action="store_true"); ap.add_argument("--rerun-completed",action="store_true"); a=ap.parse_args()
    repo=Path(__file__).resolve().parents[3]; py=repo/".venv/bin/python"; launcher=repo/"scripts/ischemia/training/run_i_s_experiment.py"; rows=[]
    for eid in DEFAULTS:
        run=repo/f"runs/I-segresnet/{eid}"; progress=run/"progress.json"
        done=False
        if progress.is_file():
            try: done=json.loads(progress.read_text()).get("status")=="completed" and (run/"final_metrics/val_probabilities").is_dir()
            except Exception: pass
        if done and not a.rerun_completed and not a.dry_run:
            print(f"[SKIP] {eid}: completed",flush=True); rows.append({"experiment":eid,"status":"skipped_completed"}); continue
        cmd=[str(py),str(launcher),"--config",str(repo/f"configs/experiments/ischemia/{eid}.yaml")]+(["--dry-run"] if a.dry_run else [])
        print(f"[START] {eid}",flush=True); code=subprocess.run(cmd,cwd=repo,check=False).returncode; rows.append({"experiment":eid,"returncode":code})
        if code: print(json.dumps(rows,ensure_ascii=False,indent=2)); return code
        print(f"[DONE] {eid}",flush=True)
    print(json.dumps(rows,ensure_ascii=False,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
