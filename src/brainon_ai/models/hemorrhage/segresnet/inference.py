from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.inferers import sliding_window_inference
from tqdm.auto import tqdm

from segresnet_baseline.metrics import binary_case_metrics, summarize_cases


def _first(value: Any):
    if isinstance(value, (list, tuple)):
        return value[0]
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.item()
    return value


def _save_prediction(prediction: torch.Tensor, image_path: str, destination: Path) -> None:
    reference = nib.load(image_path)
    array = prediction.detach().to(device="cpu", dtype=torch.uint8).numpy()
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array, reference.affine, header), str(destination))


def evaluate_loader(
    model: torch.nn.Module,
    loader,
    config: dict,
    device: torch.device,
    prediction_dir: Path | None = None,
    description: str = "validation",
) -> tuple[dict, pd.DataFrame]:
    inference = config["inference"]
    checkpoint = config["checkpoint"]
    roi_size = tuple(int(x) for x in inference["patch_size"])
    sw_batch_size = int(inference["sliding_window_batch_size"])
    overlap = float(inference["overlap"])
    mode = str(inference["blend_mode"])
    amp_enabled = bool(config["training"]["amp"] and device.type == "cuda")
    rows: list[dict] = []

    model.eval()
    with torch.inference_mode():
        for batch in tqdm(loader, desc=description, leave=False):
            # 전체 볼륨과 stitched logits는 CPU에 두고, inference patch만 GPU로 보냅니다.
            image = batch["image"]
            target = batch["label"][0, 0]
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = sliding_window_inference(
                    inputs=image,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=overlap,
                    mode=mode,
                    sw_device=device,
                    device=torch.device("cpu"),
                )
            prediction = torch.argmax(logits, dim=1)[0]
            row = binary_case_metrics(prediction, target)
            row.update(
                {
                    "source": str(_first(batch["source"])),
                    "case_id": str(_first(batch["case_id"])),
                    "group_id": str(_first(batch["group_id"])),
                    "image_path": str(_first(batch["image_path"])),
                    "label_path": str(_first(batch["label_path"])),
                }
            )
            rows.append(row)

            if prediction_dir is not None:
                filename = f"{row['source']}_{row['case_id']}.nii.gz"
                _save_prediction(prediction, row["image_path"], prediction_dir / filename)

    summary = summarize_cases(
        rows,
        dice_weight=float(checkpoint["dice_weight"]),
        recall_weight=float(checkpoint["recall_weight"]),
    )
    return summary, pd.DataFrame(rows)


def save_evaluation(summary: dict, per_case: pd.DataFrame, output_dir: Path) -> None:
    def json_safe(value):
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    output_dir.mkdir(parents=True, exist_ok=True)
    per_case.to_csv(output_dir / "per_case_metrics.csv", index=False, encoding="utf-8-sig")
    (output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
