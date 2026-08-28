from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.data import CacheDataset, DataLoader, Dataset, list_data_collate
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    ScaleIntensityRanged,
    SpatialPadd,
)
from monai.transforms.transform import RandomizableTransform
from monai.utils import MAX_SEED


class ConditionalPatchSampler(RandomizableTransform):
    """양성 volume은 병변 중심 patch를, 빈 mask는 음성 patch만 샘플링합니다."""

    def __init__(self, patch_size, num_samples: int, pos: int, neg: int):
        super().__init__(prob=1.0)
        common = dict(
            keys=["image", "label"],
            label_key="label",
            spatial_size=tuple(int(x) for x in patch_size),
            num_samples=int(num_samples),
            image_key="image",
            image_threshold=0.0,
            allow_smaller=False,
        )
        self.positive = RandCropByPosNegLabeld(pos=int(pos), neg=int(neg), **common)
        self.negative = RandCropByPosNegLabeld(pos=0, neg=1, **common)

    def set_random_state(self, seed=None, state=None):
        """MONAI worker seed를 내부의 두 crop sampler에도 전달합니다."""
        super().set_random_state(seed=seed, state=state)
        self.positive.set_random_state(seed=int(self.R.randint(MAX_SEED, dtype="uint32")))
        self.negative.set_random_state(seed=int(self.R.randint(MAX_SEED, dtype="uint32")))
        return self

    def __call__(self, data):
        if "positive_voxels" in data:
            has_foreground = int(data["positive_voxels"]) > 0
        else:
            has_foreground = bool(np.any(np.asarray(data["label"]) > 0))
        if has_foreground:
            return self.positive(data)
        # pos=0인 빈 GT에서는 MONAI의 foreground 부재 경고가 동작에 영향을 주지 않습니다.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Num foregrounds 0, Num backgrounds.*")
            return self.negative(data)


def load_manifest(path: str | Path, data_root: str | Path | None = None) -> pd.DataFrame:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"전처리 manifest가 없습니다: {manifest_path}")
    frame = pd.read_csv(manifest_path, dtype={"source": str, "case_id": str, "group_id": str})
    required = {
        "source",
        "case_id",
        "group_id",
        "split",
        "status",
        "image_path",
        "mask_path",
        "positive_voxels",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"manifest 열이 없습니다: {missing}")
    frame = frame.loc[frame.status.eq("OK")].copy()
    if len(frame) == 0:
        raise RuntimeError("status=OK인 전처리 사례가 없습니다.")
    overlap = frame.groupby(["source", "group_id"]).split.nunique().gt(1)
    if overlap.any():
        raise RuntimeError("동일 source/group_id가 둘 이상의 split에 포함됩니다.")
    expected_splits = {"train", "validation", "test"}
    actual_splits = set(frame["split"].astype(str).unique())
    if actual_splits != expected_splits:
        raise RuntimeError(f"manifest split이 예상과 다릅니다: {sorted(actual_splits)}")
    if data_root is not None:
        root = Path(data_root).resolve()
        frame["image_path"] = [
            str(root / source / "images" / f"{case_id}.nii.gz")
            for source, case_id in zip(frame.source, frame.case_id)
        ]
        frame["mask_path"] = [
            str(root / source / "masks" / f"{case_id}.nii.gz")
            for source, case_id in zip(frame.source, frame.case_id)
        ]
    missing_files = [
        value
        for column in ("image_path", "mask_path")
        for value in frame[column]
        if not Path(value).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(f"manifest 파일이 실제로 존재하지 않습니다. 첫 경로: {missing_files[0]}")
    return frame.reset_index(drop=True)


def records_for_split(frame: pd.DataFrame, split: str) -> list[dict]:
    subset = frame.loc[frame.split.eq(split)]
    return [
        {
            "image": row.image_path,
            "label": row.mask_path,
            "image_path": row.image_path,
            "label_path": row.mask_path,
            "source": row.source,
            "case_id": row.case_id,
            "group_id": row.group_id,
            "positive_voxels": int(row.positive_voxels),
        }
        for row in subset.itertuples()
    ]


def build_train_transform(config: dict):
    data = config["data"]
    low, high = (float(x) for x in data["intensity_window"])
    patch = tuple(int(x) for x in data["patch_size"])
    return Compose(
        [
            LoadImaged(keys=["image", "label"], image_only=True),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(keys=["image"], a_min=low, a_max=high, b_min=0.0, b_max=1.0, clip=True),
            SpatialPadd(keys=["image", "label"], spatial_size=patch),
            ConditionalPatchSampler(
                patch,
                num_samples=int(data["patches_per_volume"]),
                pos=int(data["positive_crop_weight"]),
                neg=int(data["negative_crop_weight"]),
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandRotate90d(keys=["image", "label"], prob=0.3, spatial_axes=(0, 1)),
            EnsureTyped(keys=["image"], dtype=torch.float32, track_meta=False),
            EnsureTyped(keys=["label"], dtype=torch.int64, track_meta=False),
        ]
    )


def build_eval_transform(config: dict):
    low, high = (float(x) for x in config["data"]["intensity_window"])
    return Compose(
        [
            LoadImaged(keys=["image", "label"], image_only=True),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(keys=["image"], a_min=low, a_max=high, b_min=0.0, b_max=1.0, clip=True),
            EnsureTyped(keys=["image"], dtype=torch.float32, track_meta=False),
            EnsureTyped(keys=["label"], dtype=torch.int64, track_meta=False),
        ]
    )


def build_loaders(config: dict, manifest: pd.DataFrame):
    data = config["data"]
    workers = int(data["num_workers"])
    cache_rate = float(data.get("cache_rate", 0.0))
    dataset_type = CacheDataset if cache_rate > 0 else Dataset
    dataset_kwargs = {"cache_rate": cache_rate, "num_workers": workers} if cache_rate > 0 else {}
    train = dataset_type(
        records_for_split(manifest, "train"),
        transform=build_train_transform(config),
        **dataset_kwargs,
    )
    validation = dataset_type(
        records_for_split(manifest, "validation"),
        transform=build_eval_transform(config),
        **dataset_kwargs,
    )
    loaders = {
        "train": DataLoader(
            train,
            batch_size=int(data["batch_size"]),
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            collate_fn=list_data_collate,
            persistent_workers=workers > 0,
        ),
        "validation": DataLoader(validation, batch_size=1, shuffle=False, num_workers=workers, pin_memory=True),
    }
    return loaders


def build_split_loader(config: dict, manifest: pd.DataFrame, split: str):
    workers = int(config["data"]["num_workers"])
    dataset = Dataset(records_for_split(manifest, split), transform=build_eval_transform(config))
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers, pin_memory=True)
