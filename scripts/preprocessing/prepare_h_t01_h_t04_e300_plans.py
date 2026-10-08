#!/usr/bin/env python3
import argparse, json, shutil
from pathlib import Path
import yaml

EXPERIMENTS = ("H-T01-E300", "H-T02-E300", "H-T03-E300", "H-T04-E300")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed-root", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    for name in EXPERIMENTS:
        cfg = yaml.safe_load((repository / f"configs/experiments/hemorrhage/tuning/{name}.yaml").read_text())
        e, plans = cfg["experiment"], cfg["plans"]
        dataset = args.preprocessed_root.resolve() / f"Dataset{int(e['dataset_id']):03d}_{e['dataset_name']}"
        source = dataset / f"{plans['base_identifier']}.json"
        destination = dataset / f"{plans['output_identifier']}.json"
        if not source.is_file():
            raise FileNotFoundError(source)
        if destination.is_file() and not args.overwrite:
            payload = json.loads(destination.read_text())
            if payload.get("plans_name") != plans["output_identifier"]:
                raise RuntimeError(f"plans_name mismatch: {destination}")
            print(f"[ok] {name}: {destination}")
            continue
        shutil.copy2(source, destination)
        payload = json.loads(destination.read_text())
        payload["plans_name"] = plans["output_identifier"]
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        print(f"[created] {name}: {destination}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
