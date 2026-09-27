#!/usr/bin/env python3
"""Dispatch final-model calibration export preparation without silently running GPU inference."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
from brainon_ai.calibration.registry import resolve_model

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--model-id",required=True); ap.add_argument("--run-gpu-export",action="store_true"); ap.add_argument("--limit",type=int); ap.add_argument("--registry",type=Path,default=Path("configs/calibration/models.yaml")); a=ap.parse_args()
    repo=Path(__file__).resolve().parents[2]; m=resolve_model(a.model_id,a.registry,repo)
    if a.model_id=="I-E03":
        cmd=[sys.executable,str(repo/"scripts/calibration/bootstrap_i_e03_calibration_exports.py")]
        if a.limit: cmd += ["--limit",str(a.limit)]
        print(" ".join(cmd)); return subprocess.run(cmd,cwd=repo,check=False).returncode
    if m.get("backend")!="nnunet": raise RuntimeError(f"No final export route registered for {a.model_id}")
    cmd=[sys.executable,str(repo/"scripts/calibration/generate_tta_probabilities.py"),"--model-id",a.model_id,"--registry",str(a.registry)]
    if a.limit: cmd += ["--limit",str(a.limit)]
    if not a.run_gpu_export:
        print("GPU export is intentionally not started. Review then run:\n"+" ".join(cmd)); return 0
    return subprocess.run(cmd,cwd=repo,check=False).returncode
if __name__=="__main__": raise SystemExit(main())
