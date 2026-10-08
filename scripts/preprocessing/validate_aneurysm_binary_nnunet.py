#!/usr/bin/env python3
"""Validate Dataset503 counts, geometry, binary labels, and source equivalence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def case_name(image_name: str) -> str:
    suffix = "_0000.nii.gz"
    if not image_name.endswith(suffix):
        raise ValueError(f"unexpected image name: {image_name}")
    return image_name[: -len(suffix)] + ".nii.gz"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8")) or {}
    source = Path(config["source_dataset"]).resolve()
    output_root = Path(config["output_root"]).resolve()
    dataset = output_root / "nnUNet_raw" / str(config["dataset_name"])
    metadata = output_root.parents[1] / "metadata"
    expected_train = int(config["expected"]["training"])
    expected_test = int(config["expected"]["test"])
    affine_tolerance = float(config.get("affine_tolerance", 0.0001))

    counts = {
        folder: len(list((dataset / folder).glob("*.nii.gz")))
        for folder in ("imagesTr", "labelsTr", "imagesTs", "test_gt")
    }
    expected_counts = {"imagesTr": expected_train, "labelsTr": expected_train, "imagesTs": expected_test, "test_gt": expected_test}
    errors: list[str] = []
    if counts != expected_counts:
        errors.append(f"count mismatch: actual={counts}, expected={expected_counts}")

    statistics: list[dict] = []
    for image_folder, label_folder in (("imagesTr", "labelsTr"), ("imagesTs", "test_gt")):
        images = {case_name(path.name): path for path in (dataset / image_folder).glob("*.nii.gz")}
        labels = {path.name: path for path in (dataset / label_folder).glob("*.nii.gz")}
        if images.keys() != labels.keys():
            errors.append(f"case mismatch in {image_folder}/{label_folder}")
        for name in sorted(images.keys() & labels.keys()):
            image = nib.load(images[name])
            label = nib.load(labels[name])
            source_label = nib.load(source / label_folder / name)
            data = np.asarray(label.dataobj)
            source_binary = np.asarray(source_label.dataobj) > 0
            values = np.unique(data)
            geometry_ok = image.shape == label.shape and np.allclose(
                image.affine, label.affine, atol=affine_tolerance, rtol=0
            )
            source_ok = (
                label.shape == source_label.shape
                and np.allclose(label.affine, source_label.affine, atol=affine_tolerance, rtol=0)
                and np.array_equal(data > 0, source_binary)
            )
            binary_ok = set(values.astype(int).tolist()).issubset({0, 1})
            if not geometry_ok:
                errors.append(f"image/label geometry mismatch: {name}")
            if not source_ok:
                errors.append(f"source foreground mismatch: {name}")
            if not binary_ok:
                errors.append(f"non-binary label: {name}, values={values.tolist()}")
            statistics.append({
                "split": "train" if label_folder == "labelsTr" else "test",
                "case_id": name.removesuffix(".nii.gz"),
                "foreground_voxels": int((data > 0).sum()),
                "empty_mask": not bool((data > 0).any()),
                "binary_ok": binary_ok,
                "geometry_ok": geometry_ok,
                "source_foreground_equal": source_ok,
                "label_sha256": sha256(labels[name]),
            })

    dataset_json = json.loads((dataset / "dataset.json").read_text(encoding="utf-8"))
    if dataset_json.get("labels") != {"background": 0, "aneurysm": 1}:
        errors.append("dataset.json labels are not binary")
    if int(dataset_json.get("numTraining", -1)) != expected_train:
        errors.append("dataset.json numTraining mismatch")

    metadata.mkdir(parents=True, exist_ok=True)
    with (metadata / "label_statistics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=statistics[0].keys())
        writer.writeheader()
        writer.writerows(statistics)
    report = {
        "status": "passed" if not errors else "failed",
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "counts": counts,
        "empty_masks": {
            split: sum(row["empty_mask"] for row in statistics if row["split"] == split)
            for split in ("train", "test")
        },
        "checked_labels": len(statistics),
        "affine_tolerance": affine_tolerance,
        "errors": errors,
    }
    (metadata / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
