#!/usr/bin/env python3
"""Build an immutable binary TopAneu nnU-Net raw dataset from Dataset501."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml


def load_config(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    required = {"source_dataset", "output_root", "dataset_id", "dataset_name", "expected"}
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"missing config keys: {missing}")
    return payload


def link_or_copy(source: Path, destination: Path, strategy: str) -> str:
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise RuntimeError(f"existing image differs in size: {destination}")
        return "existing"
    if strategy == "hardlink":
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            shutil.copy2(source, destination)
            return "copy_fallback"
    if strategy == "copy":
        shutil.copy2(source, destination)
        return "copy"
    raise ValueError(f"unsupported image_strategy: {strategy}")


def write_binary_label(source: Path, destination: Path, overwrite: bool) -> str:
    if destination.exists() and not overwrite:
        return "existing"
    image = nib.load(source)
    binary = (np.asarray(image.dataobj) > 0).astype(np.uint8)
    header = image.header.copy()
    header.set_data_dtype(np.uint8)
    temporary = destination.with_name(destination.name.removesuffix(".nii.gz") + ".tmp.nii.gz")
    nib.save(nib.Nifti1Image(binary, image.affine, header), temporary)
    os.replace(temporary, destination)
    return "binary_created"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--overwrite-labels", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    source = Path(config["source_dataset"]).resolve()
    output_root = Path(config["output_root"]).resolve()
    dataset = output_root / "nnUNet_raw" / str(config["dataset_name"])
    metadata = output_root.parents[1] / "metadata"
    strategy = str(config.get("image_strategy", "hardlink")).lower()
    if not source.is_dir():
        raise FileNotFoundError(source)

    for folder in ("imagesTr", "labelsTr", "imagesTs", "test_gt"):
        (dataset / folder).mkdir(parents=True, exist_ok=True)
    metadata.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for folder in ("imagesTr", "imagesTs"):
        for item in sorted((source / folder).glob("*.nii.gz")):
            destination = dataset / folder / item.name
            action = link_or_copy(item, destination, strategy)
            rows.append({"role": folder, "source": str(item), "destination": str(destination), "action": action, "bytes": destination.stat().st_size})

    for folder in ("labelsTr", "test_gt"):
        for item in sorted((source / folder).glob("*.nii.gz")):
            destination = dataset / folder / item.name
            action = write_binary_label(item, destination, args.overwrite_labels)
            rows.append({"role": folder, "source": str(item), "destination": str(destination), "action": action, "bytes": destination.stat().st_size})

    dataset_json = {
        "channel_names": {"0": "MRA"},
        "labels": {"background": 0, "aneurysm": 1},
        "numTraining": int(config["expected"]["training"]),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "NibabelIO",
        "name": str(config["dataset_name"]),
    }
    (dataset / "dataset.json").write_text(
        json.dumps(dataset_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with (metadata / "build_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("role", "source", "destination", "action", "bytes"))
        writer.writeheader()
        writer.writerows(rows)
    release = {
        "release_id": config.get("release_id", "topaneu_binary_v2"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(source),
        "dataset_id": int(config["dataset_id"]),
        "dataset_name": str(config["dataset_name"]),
        "binary_rule": str(config.get("binary_rule", "source_label > 0")),
        "image_strategy": strategy,
        "expected": config["expected"],
        "config": str(args.config.resolve()),
    }
    (metadata / "release.json").write_text(
        json.dumps(release, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"created: {dataset}")
    print(f"manifest: {metadata / 'build_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
