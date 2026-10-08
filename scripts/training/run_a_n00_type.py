#!/usr/bin/env python3

"""
TopAneu Type Mask Standard nnU-Net Baseline launcher.

Experiment
----------
A-N00-TYPE

Dataset
-------
Dataset502_TopAneuMRA_Type

Policy
------
- Standard nnU-Net PlainConvUNet
- 4 outputs: background + Type 1/2/3
- 3d_fullres
- fold 0
- train 215 / validation 46
- max epochs 300
- train iterations 250 / epoch
- online validation iterations 50 / epoch
- foreground oversampling 0.33
- best monitor: ema_fg_dice
- EarlyStopping start 0
- patience 30
- min_delta 0.0001
- strict resume
- final 46-patient full-volume validation once
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess


DATASET_ID = "502"

DATASET_NAME = (
    "Dataset502_TopAneuMRA_Type"
)

CONFIGURATION = "3d_fullres"

FOLD = "0"

TRAINER = (
    "nnUNetTrainerAneurysmTypeBaseline"
)

PLANS = "nnUNetPlans"


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--preprocessed",
        required=True,
        help="nnUNet_preprocessed root",
    )

    parser.add_argument(
        "--results",
        required=True,
        help="A-N00-TYPE nnUNet_results root",
    )

    parser.add_argument(
        "--python-bin",
        required=True,
        help="Python 3.12 venv executable",
    )

    parser.add_argument(
        "--raw",
        default=None,
        help="Optional nnUNet_raw root",
    )

    return parser


def main() -> int:

    args = build_parser().parse_args()

    repo = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    src_root = (
        repo
        / "src"
    )

    trainer_dir = (
        src_root
        / "brainon_ai"
        / "nnunet"
        / "trainers"
    )

    python_bin = Path(
        args.python_bin
    ).resolve()

    train_bin = (
        python_bin.parent
        / "nnUNetv2_train"
    )

    preprocessed = Path(
        args.preprocessed
    ).resolve()

    results = Path(
        args.results
    ).resolve()

    if not python_bin.is_file():

        raise FileNotFoundError(
            f"Python executable 없음: {python_bin}"
        )

    if not train_bin.is_file():

        raise FileNotFoundError(
            f"nnUNetv2_train 없음: {train_bin}"
        )

    if not trainer_dir.is_dir():

        raise FileNotFoundError(
            f"Custom Trainer directory 없음: {trainer_dir}"
        )

    dataset_preprocessed = (
        preprocessed
        / DATASET_NAME
    )

    if not dataset_preprocessed.is_dir():

        raise FileNotFoundError(
            "Dataset502 preprocessing 결과가 없습니다.\n"
            f"{dataset_preprocessed}"
        )

    env = os.environ.copy()

    env[
        "nnUNet_preprocessed"
    ] = str(
        preprocessed
    )

    env[
        "nnUNet_results"
    ] = str(
        results
    )

    if args.raw:

        env[
            "nnUNet_raw"
        ] = str(
            Path(
                args.raw
            ).resolve()
        )

    # --------------------------------------------------------
    # Custom Trainer import 경로
    #
    # repo/src가 빠지면 brainon_ai import 오류 발생 가능.
    # --------------------------------------------------------

    old_pythonpath = env.get(
        "PYTHONPATH",
        "",
    )

    pythonpath = [
        str(
            src_root
        ),
        str(
            trainer_dir
        ),
    ]

    if old_pythonpath:

        pythonpath.append(
            old_pythonpath
        )

    env[
        "PYTHONPATH"
    ] = os.pathsep.join(
        pythonpath
    )

    env[
        "nnUNet_extTrainer"
    ] = str(
        trainer_dir
    )

    # --------------------------------------------------------
    # A-N00-TYPE baseline policy
    # --------------------------------------------------------

    env[
        "A_N00_TYPE_PATIENCE"
    ] = "30"

    env[
        "A_N00_TYPE_MIN_DELTA"
    ] = "0.0001"

    # strict resume reproducibility
    env[
        "nnUNet_n_proc_DA"
    ] = "0"

    env[
        "nnUNet_compile"
    ] = "false"

    fold_dir = (
        results
        / DATASET_NAME
        / (
            "nnUNetTrainerAneurysmTypeBaseline"
            "__nnUNetPlans__3d_fullres"
        )
        / "fold_0"
    )

    checkpoint_latest = (
        fold_dir
        / "checkpoint_latest.pth"
    )

    checkpoint_final = (
        fold_dir
        / "checkpoint_final.pth"
    )

    complete_file = (
        fold_dir
        / "final_validation"
        / "COMPLETE.json"
    )

    # 이미 Final Validation까지 완료된 실험은 재실행하지 않음.
    if complete_file.is_file():

        print("=" * 80)
        print("A-N00-TYPE 이미 완료")
        print("=" * 80)

        print(
            "COMPLETE:",
            complete_file,
        )

        return 0

    resume = (
        checkpoint_latest.is_file()
        or checkpoint_final.is_file()
    )

    command = [
        str(
            train_bin
        ),

        DATASET_ID,
        CONFIGURATION,
        FOLD,

        "-tr",
        TRAINER,

        "-p",
        PLANS,

        "-device",
        "cuda",
    ]

    if resume:

        command.append(
            "--c"
        )

    print("=" * 80)
    print("TopAneu A-N00-TYPE")
    print("=" * 80)

    print(
        "Mode                  :",
        "RESUME"
        if resume
        else "NEW",
    )

    print(
        "Dataset               :",
        DATASET_NAME,
    )

    print(
        "Trainer               :",
        TRAINER,
    )

    print(
        "Architecture          : Standard PlainConvUNet",
    )

    print(
        "Outputs               : 4",
    )

    print(
        "Train / Validation    : 215 / 46",
    )

    print(
        "Max epochs            : 300",
    )

    print(
        "Train iterations      : 250",
    )

    print(
        "Online val iterations : 50",
    )

    print(
        "Foreground sampling   : 0.33",
    )

    print(
        "Best monitor          : ema_fg_dice",
    )

    print(
        "EarlyStopping         : start 0 / patience 30",
    )

    print(
        "Strict Resume         : YES",
    )

    print(
        "Final Validation      : 46 patients ONCE",
    )

    print()

    print(
        "COMMAND:"
    )

    print(
        " ".join(
            command
        )
    )

    print("=" * 80)

    result = subprocess.run(
        command,
        env=env,
    )

    return int(
        result.returncode
    )


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
