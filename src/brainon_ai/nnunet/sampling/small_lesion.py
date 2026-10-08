from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def case_weight(volume_ml: float, positive: bool, config: dict[str, Any]) -> tuple[str, float]:
    weights = config["weights"]
    if not positive:
        return "negative", float(weights["negative"])
    thresholds = config["volume_thresholds_ml"]
    if volume_ml < float(thresholds[0]):
        return "lt_1ml", float(weights["lt_1ml"])
    if volume_ml < float(thresholds[1]):
        return "1_to_5ml", float(weights["1_to_5ml"])
    if volume_ml < float(thresholds[2]):
        return "5_to_20ml", float(weights["5_to_20ml"])
    return "ge_20ml", float(weights["ge_20ml"])


def build_sampling_probabilities(dataset, spacing, config: dict[str, Any]):
    identifiers = list(dataset.identifiers)
    voxel_volume_ml = float(np.prod(np.asarray(spacing, dtype=float))) / 1000.0
    records, raw_weights = [], []
    for identifier in identifiers:
        _, segmentation, _, _ = dataset.load_case(identifier)
        positive_voxels = int(np.count_nonzero(np.asarray(segmentation) > 0))
        volume_ml = positive_voxels * voxel_volume_ml
        group, weight = case_weight(volume_ml, positive_voxels > 0, config)
        raw_weights.append(weight)
        records.append({"case_id": identifier, "positive_voxels": positive_voxels,
                        "hemorrhage_volume_ml": round(volume_ml, 6), "group": group,
                        "weight": weight})
    probabilities = np.asarray(raw_weights, dtype=np.float64)
    if probabilities.size == 0 or not np.isfinite(probabilities).all() or probabilities.sum() <= 0:
        raise ValueError("small-lesion sampling 확률을 만들 수 없습니다.")
    probabilities /= probabilities.sum()
    for record, probability in zip(records, probabilities, strict=True):
        record["probability"] = float(probability)
    group_summary = {}
    for record in records:
        item = group_summary.setdefault(str(record["group"]), {"cases": 0, "probability_mass": 0.0})
        item["cases"] += 1
        item["probability_mass"] += float(record["probability"])
    return probabilities, {"strategy": "weighted train-case sampling; validation unchanged",
                           "spacing": [float(v) for v in spacing], "voxel_volume_ml": voxel_volume_ml,
                           "num_train_cases": len(records), "group_summary": group_summary, "cases": records}


def write_sampling_report(path: str | Path, report: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)
