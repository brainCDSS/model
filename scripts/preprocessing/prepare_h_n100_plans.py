#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,shutil
from pathlib import Path
import yaml

def main():
    parser=argparse.ArgumentParser(description="H-N100 custom plans")
    parser.add_argument("--preprocessed-root",required=True,type=Path)
    parser.add_argument("--overwrite",action="store_true")
    args=parser.parse_args()
    repository=Path(__file__).resolve().parents[2]
    config=yaml.safe_load((repository/"configs/experiments/hemorrhage/H-N100.yaml").read_text())
    experiment,plans=config["experiment"],config["plans"]
    dataset=args.preprocessed_root.resolve()/f"Dataset{int(experiment['dataset_id']):03d}_{experiment['dataset_name']}"
    source=dataset/f"{plans['base_identifier']}.json"
    destination=dataset/f"{plans['output_identifier']}.json"
    if not source.is_file():raise FileNotFoundError(source)
    if destination.exists() and not args.overwrite:
        payload=json.loads(destination.read_text())
        if payload.get("plans_name")!=plans["output_identifier"]:raise RuntimeError(f"plans_name mismatch: {destination}")
        print(f"[ok] {destination}");return 0
    shutil.copy2(source,destination)
    payload=json.loads(destination.read_text());payload["plans_name"]=plans["output_identifier"]
    destination.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+chr(10))
    print(f"[created] {destination}");return 0
if __name__=="__main__":raise SystemExit(main())
