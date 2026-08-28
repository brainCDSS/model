from __future__ import annotations

import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from brainon_ai.common.checksums import sha256_file

TASKS = {
    "hemorrhage": {
        "channels": 1,
        "release": "hemorrhage_binary_v1.0.0",
        "dataset": "Dataset101_HemorrhageBinary",
    },
    "ischemia": {
        "channels": 2,
        "release": "isles2022_dwi_adc_v1.0.0",
        "dataset": "Dataset102_IschemiaDwiAdc",
    },
    "aneurysm": {
        "channels": 1,
        "release": "topaneu2026_mra_v1.0.0",
        "dataset": "Dataset103_AneurysmMra",
    },
}


def make_synthetic(
    root: Path, task: str = "all", cases: int = 4, shape: tuple[int, int, int] = (16, 16, 16)
) -> list[Path]:
    selected = TASKS if task == "all" else {task: TASKS[task]}
    manifests: list[Path] = []
    rng = np.random.default_rng(42)
    for task_name, spec in selected.items():
        target = root / task_name
        target.mkdir(parents=True, exist_ok=True)
        rows = []
        for index in range(cases):
            case_id = f"synthetic_{task_name}_{index:05d}"
            center = np.array(shape) // 2 + rng.integers(-2, 3, size=3)
            grid = np.indices(shape)
            radius = 3 if index else 0
            mask = ((grid - center[:, None, None, None]) ** 2).sum(axis=0) <= radius**2
            images = []
            for channel in range(spec["channels"]):
                image = rng.normal(0, 0.1, shape).astype(np.float32)
                image += mask.astype(np.float32) * (1.0 + channel * 0.2)
                path = target / f"{case_id}_{channel:04d}.nii.gz"
                nib.save(nib.Nifti1Image(image, np.eye(4)), path)
                images.append(str(path.resolve()))
            label_path = target / f"{case_id}.nii.gz"
            nib.save(nib.Nifti1Image(mask.astype(np.uint8), np.eye(4)), label_path)
            rows.append(
                {
                    "case_id": case_id,
                    "source_dataset": "synthetic",
                    "modality": task_name,
                    "image_paths": json.dumps(images),
                    "label_path": str(label_path.resolve()),
                    "label_positive": bool(mask.any()),
                    "shape": json.dumps(shape),
                    "spacing": json.dumps([1.0, 1.0, 1.0]),
                    "orientation": "RAS",
                    "qc_status": "passed",
                    "exclusion_reason": "",
                    "split": "train" if index < cases - 1 else "val",
                    "sha256": sha256_file(label_path),
                    "data_release": spec["release"],
                }
            )
        manifest = target / "manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (target / "READY.json").write_text(
            json.dumps({"release": spec["release"], "cases": cases}, indent=2), encoding="utf-8"
        )
        manifests.append(manifest)
    return manifests
