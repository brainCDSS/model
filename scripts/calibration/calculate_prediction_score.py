#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from brainon_ai.calibration.scoring import calculate_prediction_score

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--calibration',required=True,type=Path); ap.add_argument('--features',required=True,type=Path); ap.add_argument('--model-version'); a=ap.parse_args(); calibration=json.loads(a.calibration.read_text()); features=json.loads(a.features.read_text()); features=features.get('features',features); print(json.dumps({'prediction_score':calculate_prediction_score(features,calibration,a.model_version)},indent=2))
if __name__=='__main__': main()
