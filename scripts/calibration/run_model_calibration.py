#!/usr/bin/env python3
"""One-command TTA generation and CDSS calibration for a registered model ID."""
import argparse,os,subprocess
from pathlib import Path

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--model-id',required=True); ap.add_argument('--registry',type=Path,default=Path('configs/calibration/models.yaml')); ap.add_argument('--skip-tta',action='store_true'); ap.add_argument('--allow-provisional-target',action='store_true'); ap.add_argument('--overwrite-tta',action='store_true'); ap.add_argument('--allow-failed-model',action='store_true'); a=ap.parse_args(); repo=Path(__file__).resolve().parents[2]; py=repo/'.venv/bin/python'; env={**os.environ,'PYTHONPATH':str(repo/'src')}
 if not a.skip_tta:
  cmd=[str(py),str(repo/'scripts/calibration/generate_tta_probabilities.py'),'--model-id',a.model_id,'--registry',str(a.registry)]+(['--overwrite'] if a.overwrite_tta else []); subprocess.run(cmd,cwd=repo,env=env,check=True)
 cmd=[str(py),str(repo/'scripts/calibration/fit_prediction_score.py'),'--model-id',a.model_id,'--registry',str(a.registry)]+(['--allow-provisional-target'] if a.allow_provisional_target else [])+(['--allow-failed-model'] if a.allow_failed_model else []); return subprocess.run(cmd,cwd=repo,env=env,check=False).returncode
if __name__=='__main__': raise SystemExit(main())
