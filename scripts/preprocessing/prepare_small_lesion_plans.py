#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="H-N60/H-N70 custom plans 생성")
    parser.add_argument("--preprocessed-root", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    for name in ("H-N60", "H-N70"):
        config_path = repository / "configs/experiments/hemorrhage" / f"{name}.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        experiment, plans = config["experiment"], config["plans"]
        dataset = args.preprocessed_root.resolve() / (
            f"Dataset{int(experiment['dataset_id']):03d}_{experiment['dataset_name']}")
        source = dataset / f"{plans['base_identifier']}.json"
        destination = dataset / f"{plans['output_identifier']}.json"
        if not source.is_file():
            raise FileNotFoundError(f"base plan이 없습니다: {source}")
        if destination.exists() and not args.overwrite:
            payload = json.loads(destination.read_text(encoding="utf-8"))
            if payload.get("plans_name") != plans["output_identifier"]:
                raise RuntimeError(f"기존 plan의 plans_name이 다릅니다: {destination}")
            print(f"[ok] {name}: {destination}")
            continue
        shutil.copy2(source, destination)
        payload = json.loads(destination.read_text(encoding="utf-8"))
        payload["plans_name"] = plans["output_identifier"]
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[created] {name}: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
