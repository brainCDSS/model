#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from brainon_ai.calibration.scoring import calculate_prediction_score

def main():
    ap=argparse.ArgumentParser(description="Calculate one per-analysis segmentation reliability score")
    ap.add_argument("--score-model", "--calibration", dest="score_model", required=True, type=Path)
    ap.add_argument("--features", required=True, type=Path); ap.add_argument("--model-version")
    a=ap.parse_args(); model=json.loads(a.score_model.read_text()); payload=json.loads(a.features.read_text()); features=payload.get("features",payload)
    score=calculate_prediction_score(features,model,a.model_version)
    print(json.dumps({"prediction_score":score/100, "prediction_score_percent":round(score), "score_type":"segmentation_reliability", "score_version":model.get("score_version","MISSING")},ensure_ascii=False,indent=2))
if __name__=="__main__": main()
