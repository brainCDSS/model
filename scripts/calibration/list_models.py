#!/usr/bin/env python3
from pathlib import Path
import argparse
from brainon_ai.calibration.registry import load_registry,resolve_model

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--registry',type=Path,default=Path('configs/calibration/models.yaml')); a=ap.parse_args(); repo=Path(__file__).resolve().parents[2]
 print(f"{'MODEL ID':<10} {'TASK':<12} {'PROB':>5} {'TTA':>5}  STATUS")
 for mid in sorted(load_registry(a.registry)):
  m=resolve_model(mid,a.registry,repo); p=Path(m['prediction_dir']); t=[Path(x) for x in m['tta_prediction_dirs']]; n=len(list(p.glob('*.npz'))) if p.is_dir() else 0; nt=min([len(list(x.glob('*.npz'))) for x in t],default=0); status='READY' if n>=10 and nt==n else 'NEEDS_TTA/PROBABILITY_EXPORT'; print(f"{mid:<10} {m['task']:<12} {n:>5} {nt:>5}  {status}")
if __name__=='__main__': main()
