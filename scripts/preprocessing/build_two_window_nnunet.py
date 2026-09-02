#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

from brainon_ai.data.preprocessing import (
    apply_ct_window,
    image_with_data,
)


def load_yaml(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return yaml.safe_load(handle)


def link_or_copy(
    source: Path,
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists():
        return

    try:
        os.link(
            source,
            destination,
        )
    except OSError:
        shutil.copy2(
            source,
            destination,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Dataset001 CT에서 Brain/Subdural "
            "2채널 Dataset003을 생성합니다."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--data-root",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    config = load_yaml(args.config)
    data_root = args.data_root.resolve()

    source_config = config["source_dataset"]
    target_config = config["target_dataset"]
    windows = config["windows"]

    source_name = (
        f"Dataset"
        f"{int(source_config['dataset_id']):03d}_"
        f"{source_config['dataset_name']}"
    )

    target_name = (
        f"Dataset"
        f"{int(target_config['dataset_id']):03d}_"
        f"{target_config['dataset_name']}"
    )

    raw_root = data_root / "nnUNet_raw"
    preprocessed_root = (
        data_root / "nnUNet_preprocessed"
    )

    source_root = raw_root / source_name
    target_root = raw_root / target_name

    if not source_root.is_dir():
        raise FileNotFoundError(
            f"원본 dataset이 없습니다: {source_root}"
        )

    split_source = (
        preprocessed_root
        / source_name
        / "splits_final.json"
    )

    if not split_source.is_file():
        split_source = (
            source_root / "splits_final.json"
        )

    if not split_source.is_file():
        raise FileNotFoundError(
            f"split 파일이 없습니다: {split_source}"
        )

    selected = []

    for split_name, image_folder in [
        ("train", "imagesTr"),
        ("test", "imagesTs"),
    ]:
        for image_path in sorted(
            (source_root / image_folder).glob(
                "*_0000.nii.gz"
            )
        ):
            selected.append(
                (
                    split_name,
                    image_folder,
                    image_path,
                )
            )

    if args.max_cases is not None:
        selected = selected[
            : int(args.max_cases)
        ]

    records = []

    for index, (
        split_name,
        image_folder,
        image_path,
    ) in enumerate(selected, start=1):
        case_id = image_path.name.removesuffix(
            "_0000.nii.gz"
        )

        image = nib.load(str(image_path))

        hu = np.asarray(
            image.get_fdata(dtype=np.float32),
            dtype=np.float32,
        )

        channel_paths = []

        for window in windows:
            channel = int(window["channel"])

            windowed = apply_ct_window(
                hu,
                center=float(window["center"]),
                width=float(window["width"]),
            )

            output_path = (
                target_root
                / image_folder
                / f"{case_id}_{channel:04d}.nii.gz"
            )

            if output_path.exists():
                if not args.overwrite:
                    channel_paths.append(
                        str(output_path)
                    )
                    continue

            output_image = image_with_data(
                windowed,
                image.affine,
                np.float32,
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            nib.save(
                output_image,
                str(output_path),
            )

            channel_paths.append(
                str(output_path)
            )

        label_folder = (
            "labelsTr"
            if image_folder == "imagesTr"
            else "labelsTs"
        )

        source_label = (
            source_root
            / label_folder
            / f"{case_id}.nii.gz"
        )

        target_label = (
            target_root
            / label_folder
            / f"{case_id}.nii.gz"
        )

        if not source_label.is_file():
            raise FileNotFoundError(
                f"label이 없습니다: {source_label}"
            )

        link_or_copy(
            source_label,
            target_label,
        )

        records.append(
            {
                "case_id": case_id,
                "split": split_name,
                "channels": channel_paths,
                "label": str(target_label),
            }
        )

        if index % 25 == 0 or index == len(selected):
            print(
                f"진행: {index}/{len(selected)}"
            )

    training_cases = len(
        list(
            (target_root / "imagesTr").glob(
                "*_0000.nii.gz"
            )
        )
    )

    dataset_json = {
        "channel_names": {
            "0": "brain_window",
            "1": "subdural_window",
        },
        "labels": {
            "background": 0,
            "hemorrhage": 1,
        },
        "numTraining": training_cases,
        "file_ending": ".nii.gz",
    }

    target_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        target_root / "dataset.json"
    ).write_text(
        json.dumps(
            dataset_json,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    shutil.copy2(
        split_source,
        target_root / "splits_final.json",
    )

    run_root = (
        Path(__file__).resolve().parents[2]
        / "runs"
        / "dataset_builds"
        / target_name
    )

    run_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "source_dataset": source_name,
        "target_dataset": target_name,
        "windows": windows,
        "training_cases": training_cases,
        "total_cases": len(records),
        "target_root": str(target_root),
    }

    (
        run_root / "build_summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())