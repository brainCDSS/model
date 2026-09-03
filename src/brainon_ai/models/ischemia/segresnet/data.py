"""I-S00 DWI/ADC record and transform construction."""

from pathlib import Path

import torch
from monai.transforms import (
    Compose,
    ConcatItemsd,
    DeleteItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandFlipd,
    SpatialPadd,
)


def build_records(root: Path, split: str, include_label_path: bool = False):
    images = root / split / "images"
    labels = root / split / "labels"

    records = []

    for dwi in sorted(images.glob("*_dwi.nii.gz")):
        case_id = dwi.name.replace("_dwi.nii.gz", "")

        adc = images / f"{case_id}_adc.nii.gz"
        mask = labels / f"{case_id}_mask.nii.gz"

        if not adc.is_file():
            raise FileNotFoundError(f"ADC 없음: {case_id}")
        if not mask.is_file():
            raise FileNotFoundError(f"Mask 없음: {case_id}")

        record = {
            "case_id": case_id,
            "dwi": str(dwi),
            "adc": str(adc),
            "label": str(mask),
        }
        if include_label_path:
            record["label_path"] = str(mask)
        records.append(record)

    return records


def train_transform(patch_size, patches_per_volume):
    return Compose(
        [
            LoadImaged(keys=["dwi", "adc", "label"]),
            EnsureChannelFirstd(
                keys=["dwi", "adc", "label"],
                channel_dim="no_channel",
            ),
            # DWI + ADC → 2채널 Early Fusion
            ConcatItemsd(keys=["dwi", "adc"], name="image", dim=0),
            DeleteItemsd(keys=["dwi", "adc"]),
            # 이미 GCS 전처리 단계에서 Z-score 완료 → 재정규화하지 않음
            SpatialPadd(
                keys=["image", "label"],
                spatial_size=patch_size,
            ),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size,
                pos=1,
                neg=1,
                num_samples=patches_per_volume,
            ),
            RandFlipd(
                keys=["image", "label"],
                prob=0.5,
                spatial_axis=0,
            ),
            RandFlipd(
                keys=["image", "label"],
                prob=0.5,
                spatial_axis=1,
            ),
            EnsureTyped(keys=["image"], dtype=torch.float32),
            EnsureTyped(keys=["label"], dtype=torch.float32),
        ]
    )


def validation_transform():
    return Compose(
        [
            LoadImaged(keys=["dwi", "adc", "label"]),
            EnsureChannelFirstd(
                keys=["dwi", "adc", "label"],
                channel_dim="no_channel",
            ),
            ConcatItemsd(keys=["dwi", "adc"], name="image", dim=0),
            DeleteItemsd(keys=["dwi", "adc"]),
            EnsureTyped(keys=["image"], dtype=torch.float32),
            EnsureTyped(keys=["label"], dtype=torch.float32),
        ]
    )


def test_transform():
    """The original evaluator transform is identical to validation_transform."""
    return validation_transform()
