#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from brainon_ai.calibration.features import extract_features,foreground_probability
from brainon_ai.calibration.registry import resolve_model

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--model-id',required=True); ap.add_argument('--case-id',required=True); ap.add_argument('--registry',type=Path,default=Path('configs/calibration/models.yaml')); ap.add_argument('--output',type=Path); a=ap.parse_args(); repo=Path(__file__).resolve().parents[2]; m=resolve_model(a.model_id,a.registry,repo); original=foreground_probability(Path(m['prediction_dir'])/f'{a.case_id}.npz'); tta=[foreground_probability(Path(d)/f'{a.case_id}.npz') for d in m['tta_prediction_dirs']]; result={'model_id':a.model_id,'model_version':m['version'],'case_id':a.case_id,'features':extract_features(original,float(m['threshold']),tta,int(m.get('connectivity',26)))}; text=json.dumps(result,ensure_ascii=False,indent=2)+'\n'; print(text,end='');
 if a.output: a.output.write_text(text,encoding='utf-8')
if __name__=='__main__': main()
