#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
from sklearn.model_selection import StratifiedKFold


DATASET_NAME = "Dataset001_ICHBinary"
RANDOM_SEED = 42
TOTAL_FOLDS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "기존 fold 0을 유지하면서 "
            "H-T01-E300용 fold 1~4를 생성합니다."
        )
    )

    parser.add_argument(
        "--preprocessed-root",
        required=True,
        type=Path,
        help="nnUNet_preprocessed 루트",
    )

    parser.add_argument(
        "--write",
        action="store_true",
        help="검증 후 실제 splits_final.json을 수정",
    )

    return parser.parse_args()


def get_source(case_id: str) -> str:
    """case ID에서 데이터 출처를 구분합니다."""

    upper = case_id.upper()

    if upper.startswith("BHSD"):
        return "BHSD"

    if upper.startswith("PHYS"):
        return "PhysioNet"

    return "unknown"


def is_positive_case(
    case_id: str,
    gt_folder: Path,
) -> bool:
    """GT에 foreground voxel이 하나라도 있는지 확인합니다."""

    mask_path = gt_folder / f"{case_id}.nii.gz"

    if not mask_path.is_file():
        raise FileNotFoundError(
            f"GT mask가 없습니다: {mask_path}"
        )

    mask = np.asanyarray(
        nib.load(mask_path).dataobj
    )

    return bool(np.any(mask > 0))


def describe_fold(
    fold_index: int,
    split: dict,
    strata: dict[str, str],
) -> None:
    """fold별 case 수와 계층 분포를 출력합니다."""

    train = list(split["train"])
    val = list(split["val"])

    train_counter = Counter(
        strata[case_id]
        for case_id in train
    )

    val_counter = Counter(
        strata[case_id]
        for case_id in val
    )

    print()
    print(f"Fold {fold_index}")
    print("-" * 60)
    print("Train      :", len(train))
    print("Validation :", len(val))
    print(
        "Overlap    :",
        len(set(train) & set(val)),
    )
    print("Train strata:", dict(train_counter))
    print("Val strata  :", dict(val_counter))


def validate_splits(
    splits: list[dict],
    all_cases: set[str],
    original_fold0_val: list[str],
) -> None:
    """K-Fold 구성이 올바른지 검증합니다."""

    if len(splits) != TOTAL_FOLDS:
        raise ValueError(
            f"fold 수가 5가 아닙니다: {len(splits)}"
        )

    if splits[0]["val"] != original_fold0_val:
        raise ValueError(
            "기존 fold 0 validation이 변경됐습니다."
        )

    validation_cases = []

    for fold_index, split in enumerate(splits):
        train = set(split["train"])
        val = set(split["val"])

        if train & val:
            raise ValueError(
                f"fold {fold_index}: "
                "train/validation 중복이 있습니다."
            )

        if train | val != all_cases:
            missing = all_cases - (train | val)
            unknown = (train | val) - all_cases

            raise ValueError(
                f"fold {fold_index}: case 구성이 다릅니다. "
                f"missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )

        validation_cases.extend(split["val"])

    validation_counter = Counter(validation_cases)

    duplicated = sorted(
        case_id
        for case_id, count
        in validation_counter.items()
        if count != 1
    )

    if duplicated:
        raise ValueError(
            "validation에 한 번이 아니게 등장한 case가 있습니다: "
            f"{duplicated}"
        )

    if set(validation_cases) != all_cases:
        missing = all_cases - set(validation_cases)

        raise ValueError(
            "어떤 fold의 validation에도 포함되지 않은 case가 있습니다: "
            f"{sorted(missing)}"
        )


def main() -> int:
    args = parse_args()

    dataset_folder = (
        args.preprocessed_root.resolve()
        / DATASET_NAME
    )

    split_path = (
        dataset_folder
        / "splits_final.json"
    )

    backup_path = (
        dataset_folder
        / "splits_final_fold0_backup.json"
    )

    gt_folder = (
        dataset_folder
        / "gt_segmentations"
    )

    if not split_path.is_file():
        raise FileNotFoundError(
            f"splits_final.json이 없습니다: {split_path}"
        )

    if not gt_folder.is_dir():
        raise FileNotFoundError(
            f"GT 폴더가 없습니다: {gt_folder}"
        )

    existing_splits = json.loads(
        split_path.read_text(encoding="utf-8")
    )

    if not existing_splits:
        raise ValueError(
            "splits_final.json이 비어 있습니다."
        )

    # ---------------------------------------------------------
    # 기존 fold 0은 절대 변경하지 않습니다.
    # ---------------------------------------------------------

    fold0 = {
        "train": list(existing_splits[0]["train"]),
        "val": list(existing_splits[0]["val"]),
    }

    original_fold0_val = list(fold0["val"])

    all_cases = set(fold0["train"]) | set(fold0["val"])

    if set(fold0["train"]) & set(fold0["val"]):
        raise ValueError(
            "기존 fold 0에 train/validation 중복이 있습니다."
        )

    print("전체 case:", len(all_cases))
    print("Fold 0 train:", len(fold0["train"]))
    print("Fold 0 validation:", len(fold0["val"]))

    # ---------------------------------------------------------
    # 출처와 양성/음성을 합친 stratification label 생성
    # 예: BHSD_positive, PhysioNet_negative
    # ---------------------------------------------------------

    strata: dict[str, str] = {}

    for index, case_id in enumerate(sorted(all_cases), start=1):
        source = get_source(case_id)

        positive = is_positive_case(
            case_id,
            gt_folder,
        )

        label_status = (
            "positive"
            if positive
            else "negative"
        )

        strata[case_id] = (
            f"{source}_{label_status}"
        )

        if index % 25 == 0:
            print(
                f"GT 확인: {index}/{len(all_cases)}"
            )

    print()
    print(
        "전체 strata:",
        dict(Counter(strata.values())),
    )

    # ---------------------------------------------------------
    # fold 0 train 186명을 4개 validation 그룹으로 분할
    #
    # fold 0 validation 40명은 그대로 보존합니다.
    # ---------------------------------------------------------

    remaining_cases = np.array(
        sorted(fold0["train"])
    )

    remaining_strata = np.array(
        [
            strata[case_id]
            for case_id in remaining_cases
        ]
    )

    stratum_counts = Counter(remaining_strata)

    too_small = {
        name: count
        for name, count in stratum_counts.items()
        if count < 4
    }

    if too_small:
        raise ValueError(
            "4-way stratification을 하기에는 "
            "case가 부족한 strata가 있습니다: "
            f"{too_small}"
        )

    splitter = StratifiedKFold(
        n_splits=4,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    new_splits = [fold0]

    for _, validation_indices in splitter.split(
        remaining_cases,
        remaining_strata,
    ):
        val_cases = sorted(
            remaining_cases[
                validation_indices
            ].tolist()
        )

        val_set = set(val_cases)

        train_cases = sorted(
            all_cases - val_set
        )

        new_splits.append(
            {
                "train": train_cases,
                "val": val_cases,
            }
        )

    # ---------------------------------------------------------
    # 생성 결과 검증
    # ---------------------------------------------------------

    validate_splits(
        splits=new_splits,
        all_cases=all_cases,
        original_fold0_val=original_fold0_val,
    )

    for fold_index, split in enumerate(new_splits):
        describe_fold(
            fold_index=fold_index,
            split=split,
            strata=strata,
        )

    print()
    print("=" * 60)
    print("K-Fold 검증 통과")
    print("전체 fold 수:", len(new_splits))
    print(
        "validation 고유 case 수:",
        len(
            {
                case_id
                for split in new_splits
                for case_id in split["val"]
            }
        ),
    )
    print("=" * 60)

    # ---------------------------------------------------------
    # --write가 없으면 파일을 수정하지 않습니다.
    # ---------------------------------------------------------

    if not args.write:
        print()
        print(
            "Dry-run만 완료했습니다. "
            "실제 저장하려면 --write를 추가하세요."
        )

        return 0

    # 최초 1회만 기존 파일 백업
    if not backup_path.exists():
        shutil.copy2(
            split_path,
            backup_path,
        )

        print(
            "기존 split 백업:",
            backup_path,
        )

    temporary_path = split_path.with_suffix(
        ".json.tmp"
    )

    temporary_path.write_text(
        json.dumps(
            new_splits,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(split_path)

    print(
        "새 split 저장:",
        split_path,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())