#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import yaml


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def link_or_copy(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        return

    if mode == "hardlink":
        os.link(source, destination)
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError(f"지원하지 않는 copy_mode: {mode}")


def filter_case_ids(case_ids: list[str], prefix: str) -> list[str]:
    return sorted(case_id for case_id in case_ids if case_id.startswith(prefix))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dataset001에서 BHSD case만 골라 Dataset002를 생성합니다."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    args = parser.parse_args()

    config = load_yaml(args.config)
    data_root = args.data_root.resolve()

    source = config["source_dataset"]
    target = config["target_dataset"]
    prefix = str(config["case_prefix"])
    copy_mode = str(config.get("copy_mode", "hardlink"))

    raw_root = data_root / "nnUNet_raw"
    preprocessed_root = data_root / "nnUNet_preprocessed"

    source_name = (
        f"Dataset{int(source['dataset_id']):03d}_"
        f"{source['dataset_name']}"
    )
    target_name = (
        f"Dataset{int(target['dataset_id']):03d}_"
        f"{target['dataset_name']}"
    )

    source_root = raw_root / source_name
    target_root = raw_root / target_name

    if not source_root.is_dir():
        raise FileNotFoundError(f"원본 dataset이 없습니다: {source_root}")

    source_split_path = (
        preprocessed_root / source_name / "splits_final.json"
    )
    if not source_split_path.is_file():
        source_split_path = source_root / "splits_final.json"

    if not source_split_path.is_file():
        raise FileNotFoundError(
            f"splits_final.json이 없습니다: {source_split_path}"
        )

    split = json.loads(source_split_path.read_text(encoding="utf-8"))[0]

    filtered_split = [
        {
            "train": filter_case_ids(split["train"], prefix),
            "val": filter_case_ids(split["val"], prefix),
        }
    ]

    copied = []

    for directory in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        source_directory = source_root / directory

        if not source_directory.is_dir():
            continue

        for source_file in sorted(source_directory.glob(f"{prefix}*.nii.gz")):
            destination = target_root / directory / source_file.name
            link_or_copy(source_file, destination, copy_mode)
            copied.append(
                {
                    "directory": directory,
                    "file": source_file.name,
                    "source": str(source_file),
                    "destination": str(destination),
                }
            )

    training_cases = len(
        list((target_root / "imagesTr").glob("*_0000.nii.gz"))
    )

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {
            "background": 0,
            "hemorrhage": 1,
        },
        "numTraining": training_cases,
        "file_ending": ".nii.gz",
    }

    target_root.mkdir(parents=True, exist_ok=True)

    (target_root / "dataset.json").write_text(
        json.dumps(dataset_json, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (target_root / "splits_final.json").write_text(
        json.dumps(filtered_split, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    run_root = (
        Path(__file__).resolve().parents[2]
        / "runs"
        / "dataset_builds"
        / target_name
    )
    run_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "source_dataset": source_name,
        "target_dataset": target_name,
        "copy_mode": copy_mode,
        "train_cases": len(filtered_split[0]["train"]),
        "validation_cases": len(filtered_split[0]["val"]),
        "linked_files": len(copied),
        "target_root": str(target_root),
    }

    (run_root / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_root / "resolved_split.json").write_text(
        json.dumps(filtered_split, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())