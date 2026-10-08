#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(
        description="H-N30용 기본 nnU-Net Attention plan을 생성합니다."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--preprocessed-root",
        required=True,
        type=Path,
    )
    args = parser.parse_args()

    config = yaml.safe_load(
        args.config.read_text(encoding="utf-8")
    )

    experiment = config["experiment"]
    plans = config["plans"]

    dataset_id = int(experiment["dataset_id"])
    dataset_name = str(
        experiment.get("dataset_name", "ICHBinary")
    )

    dataset_folder = (
        f"Dataset{dataset_id:03d}_{dataset_name}"
    )

    dataset_root = (
        args.preprocessed_root.resolve()
        / dataset_folder
    )

    base_identifier = str(plans["base_identifier"])
    output_identifier = str(plans["output_identifier"])

    source = dataset_root / f"{base_identifier}.json"
    destination = dataset_root / f"{output_identifier}.json"

    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"전처리 dataset 폴더가 없습니다: {dataset_root}"
        )

    if not source.is_file():
        raise FileNotFoundError(
            f"기본 plan 파일이 없습니다: {source}"
        )

    if destination.exists():
        raise FileExistsError(
            f"H-N30 plan이 이미 존재합니다: {destination}"
        )

    shutil.copy2(source, destination)

    payload = json.loads(
        destination.read_text(encoding="utf-8")
    )
    payload["plans_name"] = output_identifier

    destination.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"base plan: {source}")
    print(f"H-N30 plan: {destination}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())