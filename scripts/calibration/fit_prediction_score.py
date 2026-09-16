#!/usr/bin/env python3
from pathlib import Path
import argparse,json,yaml
from brainon_ai.calibration.pipeline import fit_calibration
from brainon_ai.calibration.registry import resolve_model

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--model-id',required=True); ap.add_argument('--registry',type=Path,default=Path('configs/calibration/models.yaml')); ap.add_argument('--target-config',type=Path); ap.add_argument('--output-dir',type=Path); ap.add_argument('--allow-provisional-target',action='store_true'); ap.add_argument('--allow-failed-model',action='store_true'); a=ap.parse_args(); repo=Path(__file__).resolve().parents[2]; m=resolve_model(a.model_id,a.registry,repo);
 if m.get('calibration_eligible') is False and not a.allow_failed_model: raise RuntimeError(f"{a.model_id} is not calibration-eligible: {m.get('model_status')}")
 tc=a.target_config or repo/f"configs/calibration/tasks/{m['task']}.yaml"; target=yaml.safe_load(Path(tc).read_text()); out=a.output_dir or repo/f"artifacts/calibration/{m['task']}/{a.model_id}/{m['version']}"; result=fit_calibration(m,target,Path(out),a.allow_provisional_target); print(json.dumps({'output_dir':str(out),'metrics':result['metrics'],'training':result['training']},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
