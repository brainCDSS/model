#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(
        description="H-N50 2채널 nnU-Net용 custom plans를 생성합니다."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--preprocessed-root", required=True, type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="기존 H-N50 plan을 같은 설정으로 다시 생성합니다.",
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    experiment = config["experiment"]
    plans = config["plans"]

    dataset_folder = (
        f"Dataset{int(experiment['dataset_id']):03d}_"
        f"{experiment['dataset_name']}"
    )
    dataset_root = args.preprocessed_root.resolve() / dataset_folder
    source = dataset_root / f"{plans['base_identifier']}.json"
    destination = dataset_root / f"{plans['output_identifier']}.json"

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"전처리 dataset 폴더가 없습니다: {dataset_root}")
    if not source.is_file():
        raise FileNotFoundError(
            "기본 plan이 없습니다. 먼저 Dataset003 planning/preprocessing을 "
            f"완료하세요: {source}"
        )
    if destination.exists() and not args.overwrite:
        raise FileExistsError(
            f"H-N50 plan이 이미 존재합니다: {destination}\n"
            "다시 만들려면 --overwrite를 사용하세요."
        )

    shutil.copy2(source, destination)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    payload["plans_name"] = str(plans["output_identifier"])
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"base plan: {source}")
    print(f"H-N50 plan: {destination}")
    print(f"plans_name: {payload['plans_name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
