#!/usr/bin/env python3
import argparse,json,shutil
from pathlib import Path
import yaml
EXPERIMENTS=("H-T05","H-T06","H-T07","H-T08")
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--preprocessed-root",required=True,type=Path);parser.add_argument("--overwrite",action="store_true");args=parser.parse_args()
    repository=Path(__file__).resolve().parents[2]
    for name in EXPERIMENTS:
        cfg=yaml.safe_load((repository/f"configs/experiments/hemorrhage/tuning/{name}.yaml").read_text());e,p=cfg["experiment"],cfg["plans"]
        dataset=args.preprocessed_root.resolve()/f"Dataset{int(e['dataset_id']):03d}_{e['dataset_name']}";source=dataset/f"{p['base_identifier']}.json";destination=dataset/f"{p['output_identifier']}.json"
        if not source.is_file():raise FileNotFoundError(source)
        if destination.exists() and not args.overwrite:
            payload=json.loads(destination.read_text())
            if payload.get("plans_name")!=p["output_identifier"]:raise RuntimeError(f"plans_name mismatch: {destination}")
            print(f"[ok] {name}: {destination}");continue
        shutil.copy2(source,destination);payload=json.loads(destination.read_text());payload["plans_name"]=p["output_identifier"]
        destination.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+chr(10));print(f"[created] {name}: {destination}")
    return 0
if __name__=="__main__":raise SystemExit(main())
