from __future__ import annotations

import hashlib
import json
import warnings

import nibabel as nib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.data import Dataset, list_data_collate
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    SpatialPadd,
)
from monai.transforms.transform import RandomizableTransform
from monai.utils import MAX_SEED

from brainon_ai.models.aneurysm.segresnet.config import (
    resolve_repository_path,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


class ConditionalPatchSampler(RandomizableTransform):
    """Sample valid patches from positive and negative patients."""

    def __init__(
        self,
        patch_size,
        foreground_fraction: float,
    ):
        super().__init__(prob=1.0)

        fraction = float(foreground_fraction)

        if not 0.0 <= fraction <= 1.0:
            raise ValueError(
                "foreground_oversample_fraction "
                "must be in [0, 1]"
            )

        common = {
            "keys": ["image", "label"],
            "label_key": "label",
            "spatial_size": tuple(
                int(value)
                for value in patch_size
            ),
            "num_samples": 1,
            "image_key": "image",
            "image_threshold": 0.0,
            "allow_smaller": False,
        }

        scale = 10_000
        positive_weight = int(
            round(fraction * scale)
        )
        negative_weight = (
            scale - positive_weight
        )

        self.mixed = RandCropByPosNegLabeld(
            pos=positive_weight,
            neg=negative_weight,
            **common,
        )

        self.background_only = (
            RandCropByPosNegLabeld(
                pos=0,
                neg=1,
                **common,
            )
        )

    def set_random_state(
        self,
        seed=None,
        state=None,
    ):
        super().set_random_state(
            seed=seed,
            state=state,
        )

        self.mixed.set_random_state(
            seed=int(
                self.R.randint(
                    MAX_SEED,
                    dtype="uint32",
                )
            )
        )

        self.background_only.set_random_state(
            seed=int(
                self.R.randint(
                    MAX_SEED,
                    dtype="uint32",
                )
            )
        )

        return self

    def __call__(self, data):
        label = np.asarray(data["label"])
        has_foreground = bool(
            np.any(label > 0)
        )

        sampler = (
            self.mixed
            if has_foreground
            else self.background_only
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "Num foregrounds 0, "
                    "Num backgrounds.*"
                ),
            )
            return sampler(data)


def load_manifest(config: dict) -> pd.DataFrame:
    data = config["data"]

    manifest_path = resolve_repository_path(
        data["manifest_path"]
    )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            "A-S00 manifest does not exist: "
            f"{manifest_path}"
        )

    split_info_path = resolve_repository_path(
        data["split_info_path"]
    )

    if not split_info_path.is_file():
        raise FileNotFoundError(
            "A-S00 split_info.json does not exist: "
            f"{split_info_path}"
        )

    actual_split_hash = sha256_file(
        split_info_path
    )

    if actual_split_hash != str(
        data["split_info_sha256"]
    ):
        raise RuntimeError(
            "split_info.json SHA-256 does not "
            "match the frozen split contract"
        )

    split_info = json.loads(
        split_info_path.read_text(
            encoding="utf-8-sig"
        )
    )

    expected_split_info = {
        "seed": 20260828,
        "train": 215,
        "validation": 46,
        "test": 46,
        "patient_overlap": 0,
        "shared_across_models": True,
        "shared_across_targets": True,
        "test_used_for_fingerprint": False,
        "test_used_for_planning": False,
        "test_used_for_training_preprocessing": False,
    }

    for key, expected_value in (
        expected_split_info.items()
    ):
        actual_value = split_info.get(key)

        if actual_value != expected_value:
            raise RuntimeError(
                "Frozen split contract mismatch: "
                f"{key}={actual_value!r}, "
                f"expected={expected_value!r}"
            )

    common_manifest_path = (
        resolve_repository_path(
            data["common_manifest_path"]
        )
    )

    if not common_manifest_path.is_file():
        raise FileNotFoundError(
            "Common dataset manifest does not exist: "
            f"{common_manifest_path}"
        )

    actual_common_hash = sha256_file(
        common_manifest_path
    )

    if actual_common_hash != str(
        data["common_manifest_sha256"]
    ):
        raise RuntimeError(
            "Common dataset manifest SHA-256 "
            "does not match the contract"
        )

    common = pd.read_csv(
        common_manifest_path,
        encoding="utf-8-sig",
        dtype=str,
    )

    common.columns = [
        str(column).strip()
        for column in common.columns
    ]

    if not {
        "case_id",
        "split",
    }.issubset(common.columns):
        raise ValueError(
            "Common dataset manifest requires "
            "case_id and split columns"
        )

    common["case_id"] = (
        common["case_id"]
        .astype(str)
        .str.strip()
    )

    common["split"] = (
        common["split"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    common_counts = (
        common.groupby("split")
        .size()
        .to_dict()
    )

    expected_common_counts = {
        "test": 46,
        "train": 215,
        "validation": 46,
    }

    if common_counts != expected_common_counts:
        raise RuntimeError(
            "Common frozen split counts are invalid: "
            f"{common_counts}"
        )

    frame = pd.read_csv(
        manifest_path,
        encoding="utf-8-sig",
        dtype=str,
    )

    frame.columns = [
        str(column).strip()
        for column in frame.columns
    ]

    required = {
        "case_id",
        "split",
        "image_path",
        "label_path",
        "location_json_path",
    }

    missing = sorted(
        required - set(frame.columns)
    )

    if missing:
        raise ValueError(
            "A-S00 manifest columns are missing: "
            f"{missing}"
        )

    for column in required:
        frame[column] = (
            frame[column]
            .astype(str)
            .str.strip()
        )

    frame["split"] = (
        frame["split"].str.lower()
    )

    if frame["case_id"].duplicated().any():
        raise ValueError(
            "A-S00 manifest contains "
            "duplicate case_id values"
        )

    counts = (
        frame.groupby("split")
        .size()
        .to_dict()
    )

    expected_counts = {
        "train": int(
            data["expected_train_cases"]
        ),
        "validation": int(
            data["expected_validation_cases"]
        ),
    }

    if counts != expected_counts:
        raise ValueError(
            f"A-S00 split must be "
            f"{expected_counts}, got {counts}"
        )

    if "test" in set(frame["split"]):
        raise ValueError(
            "A-S00 development manifest must "
            "not contain test cases"
        )

    if int(
        data["expected_manifest_test_cases"]
    ) != 0:
        raise ValueError(
            "A-S00 config must keep "
            "development test count at zero"
        )

    expected_development_ids = set(
        common.loc[
            common["split"].isin(
                ["train", "validation"]
            ),
            "case_id",
        ]
    )

    actual_development_ids = set(
        frame["case_id"]
    )

    if (
        actual_development_ids
        != expected_development_ids
    ):
        raise RuntimeError(
            "A-S00 case IDs differ from the "
            "common frozen train/validation split"
        )

    root = resolve_repository_path(
        data["root_dir"]
    )

    expected_parents = {
        "image_path": (
            root / "images"
        ).resolve(),
        "label_path": (
            root / "location_masks"
        ).resolve(),
        "location_json_path": (
            root / "location_jsons"
        ).resolve(),
    }

    for column, expected_parent in (
        expected_parents.items()
    ):
        for raw_path in frame[column]:
            path = Path(raw_path).resolve()

            if path.parent != expected_parent:
                raise ValueError(
                    f"Unexpected {column} "
                    f"location: {path}"
                )

            if (
                not path.is_file()
                or path.stat().st_size <= 0
            ):
                raise FileNotFoundError(
                    "Missing or empty A-S00 input: "
                    f"{path}"
                )

    forbidden_tokens = (
        "Dataset501",
        "Dataset502",
        "nnUNet_preprocessed",
        "type_masks",
    )

    serialized_rows = (
        frame.astype(str)
        .agg(" ".join, axis=1)
    )

    for token in forbidden_tokens:
        if serialized_rows.str.contains(
            token,
            case=False,
            regex=False,
        ).any():
            raise ValueError(
                "A-S00 manifest contains "
                f"forbidden data reference: {token}"
            )

    return (
        frame.sort_values("case_id")
        .reset_index(drop=True)
    )


def records_for_split(
    frame: pd.DataFrame,
    split: str,
) -> list[dict]:
    subset = frame.loc[
        frame["split"].eq(split)
    ]

    return [
        {
            "image": row.image_path,
            "label": row.label_path,
            "image_path": row.image_path,
            "label_path": row.label_path,
            "location_json_path": (
                row.location_json_path
            ),
            "case_id": row.case_id,
            "split": row.split,
        }
        for row in subset.itertuples(index=False)
    ]


def _base_transform(config: dict) -> list:
    normalization = str(
        config["data"]["normalization"]
    ).lower()

    if normalization != "nonzero_zscore":
        raise ValueError(
            "Unsupported A-S00 normalization: "
            f"{normalization}"
        )

    return [
        LoadImaged(
            keys=["image", "label"],
            image_only=True,
        ),
        EnsureChannelFirstd(
            keys=["image", "label"]
        ),
        NormalizeIntensityd(
            keys=["image"],
            nonzero=True,
            channel_wise=True,
        ),
    ]


def build_patch_transform(
    config: dict,
    *,
    training: bool,
    foreground_fraction: float | None = None,
) -> Compose:
    data = config["data"]

    fraction = (
        float(data["foreground_oversample_fraction"])
        if foreground_fraction is None
        else float(foreground_fraction)
    )

    transforms = _base_transform(config)
    transforms.extend(
        [
            SpatialPadd(
                keys=["image", "label"],
                spatial_size=data["patch_size"],
            ),
            ConditionalPatchSampler(
                data["patch_size"],
                foreground_fraction=fraction,
            ),
        ]
    )

    if training:
        transforms.extend(
            [
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
                RandFlipd(
                    keys=["image", "label"],
                    prob=0.5,
                    spatial_axis=2,
                ),
                RandRotate90d(
                    keys=["image", "label"],
                    prob=0.3,
                    spatial_axes=(0, 1),
                ),
                RandScaleIntensityd(
                    keys=["image"],
                    factors=0.1,
                    prob=0.15,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.1,
                    prob=0.15,
                ),
            ]
        )

    transforms.extend(
        [
            EnsureTyped(
                keys=["image"],
                dtype=torch.float32,
                track_meta=False,
            ),
            EnsureTyped(
                keys=["label"],
                dtype=torch.int64,
                track_meta=False,
            ),
        ]
    )

    return Compose(transforms)

def build_full_volume_transform(
    config: dict,
) -> Compose:
    return Compose(
        _base_transform(config)
        + [
            EnsureTyped(
                keys=["image"],
                dtype=torch.float32,
                track_meta=False,
            ),
            EnsureTyped(
                keys=["label"],
                dtype=torch.int64,
                track_meta=False,
            ),
        ]
    )


def _records_with_foreground(
    records: list[dict],
) -> list[dict]:
    positive_records = []

    for record in records:
        label_image = nib.load(record["label"])
        label_array = np.asanyarray(
            label_image.dataobj
        )

        if np.any(label_array > 0):
            positive_records.append(record)

    if not positive_records:
        raise RuntimeError(
            "Split contains no foreground-positive cases"
        )

    return positive_records


def build_patch_datasets(
    config: dict,
    manifest: pd.DataFrame,
) -> dict[str, Dataset]:
    train_records = records_for_split(
        manifest,
        "train",
    )
    validation_records = records_for_split(
        manifest,
        "validation",
    )

    train_positive = _records_with_foreground(
        train_records
    )
    validation_positive = _records_with_foreground(
        validation_records
    )

    return {
        "train_foreground": Dataset(
            train_positive,
            transform=build_patch_transform(
                config,
                training=True,
                foreground_fraction=1.0,
            ),
        ),
        "train_background": Dataset(
            train_records,
            transform=build_patch_transform(
                config,
                training=True,
                foreground_fraction=0.0,
            ),
        ),
        "validation_foreground": Dataset(
            validation_positive,
            transform=build_patch_transform(
                config,
                training=False,
                foreground_fraction=1.0,
            ),
        ),
        "validation_background": Dataset(
            validation_records,
            transform=build_patch_transform(
                config,
                training=False,
                foreground_fraction=0.0,
            ),
        ),
    }


def deterministic_foreground_unit(
    *,
    iteration: int,
    total_iterations: int,
    fraction: float,
) -> bool:
    """Return an exact, interleaved forced-foreground schedule."""
    if total_iterations <= 0:
        raise ValueError(
            "total_iterations must be positive"
        )

    if not 0.0 <= fraction <= 1.0:
        raise ValueError(
            "fraction must be in [0, 1]"
        )

    if not 0 <= iteration < total_iterations:
        raise ValueError(
            "iteration is out of range"
        )

    foreground_count = int(
        round(total_iterations * fraction)
    )

    count_before = (
        iteration * foreground_count
    ) // total_iterations

    count_after = (
        (iteration + 1) * foreground_count
    ) // total_iterations

    return count_after > count_before

def deterministic_indices(
    *,
    case_count: int,
    batch_size: int,
    seed: int,
    epoch: int,
    iteration: int,
    phase: str,
) -> list[int]:
    """Return the deterministic case indices for one unit."""

    if phase not in {
        "train",
        "online_validation",
    }:
        raise ValueError(
            f"Unsupported patch phase: {phase}"
        )

    if case_count <= 0:
        raise ValueError(
            "case_count must be positive"
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive"
        )

    phase_offset = (
        0
        if phase == "train"
        else 1_000_000_007
    )

    sequence = np.random.SeedSequence(
        [
            int(seed),
            int(epoch),
            int(iteration),
            phase_offset,
        ]
    )

    generator = np.random.default_rng(
        sequence
    )

    return generator.integers(
        0,
        case_count,
        size=batch_size,
    ).tolist()


def deterministic_batch(
    dataset: Dataset,
    *,
    indices: list[int],
    transform_seed: int,
) -> dict:
    if not indices:
        raise ValueError(
            "indices must not be empty"
        )

    if hasattr(
        dataset.transform,
        "set_random_state",
    ):
        dataset.transform.set_random_state(
            seed=int(transform_seed) % MAX_SEED
        )

    samples = [
        dataset[index]
        for index in indices
    ]

    return list_data_collate(samples)
