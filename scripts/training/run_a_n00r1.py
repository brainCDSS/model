#!/usr/bin/env python3

"""
TopAneu A-N00R1 corrected retrain launcher.

실험:
- Dataset501
- 3d_fullres
- fold 0
- nnUNetTrainerAneurysmSparse53Retrain

정책:
- GT-present-only Dice + CE
- foreground oversampling 0.75
- EarlyStopping start 80
- patience 40
- min_delta 0.0001
- strict resume inherited from A-N00
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess


DATASET_ID = "501"
CONFIGURATION = "3d_fullres"
FOLD = "0"
TRAINER = "nnUNetTrainerAneurysmSparse53Retrain"
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
        help="A-N00R1 nnUNet_results root",
    )

    parser.add_argument(
        "--python-bin",
        required=True,
        help="A-N00R1 Python executable",
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

    if not python_bin.is_file():
        raise FileNotFoundError(
            f"Python 없음: {python_bin}"
        )

    if not train_bin.is_file():
        raise FileNotFoundError(
            f"nnUNetv2_train 없음: {train_bin}"
        )

    if not trainer_dir.is_dir():
        raise FileNotFoundError(
            f"Trainer directory 없음: {trainer_dir}"
        )

    env = os.environ.copy()

    env["nnUNet_preprocessed"] = str(
        Path(
            args.preprocessed
        ).resolve()
    )

    env["nnUNet_results"] = str(
        Path(
            args.results
        ).resolve()
    )

    if args.raw:
        env["nnUNet_raw"] = str(
            Path(
                args.raw
            ).resolve()
        )

    # --------------------------------------------------------
    # Custom trainer import
    #
    # brainon_ai package를 import하려면
    # repo/src가 반드시 PYTHONPATH에 있어야 함.
    # --------------------------------------------------------

    old_pythonpath = env.get(
        "PYTHONPATH",
        ""
    )

    pythonpath_parts = [
        str(src_root),
        str(trainer_dir),
    ]

    if old_pythonpath:
        pythonpath_parts.append(
            old_pythonpath
        )

    env["PYTHONPATH"] = (
        os.pathsep.join(
            pythonpath_parts
        )
    )

    env["nnUNet_extTrainer"] = str(
        trainer_dir
    )

    # --------------------------------------------------------
    # A-N00R1 policy
    # --------------------------------------------------------

    env["A_N00R1_OVERSAMPLE"] = "0.75"

    env[
        "A_N00R1_EARLY_STOP_START"
    ] = "80"

    env["A_N00R1_PATIENCE"] = "40"

    env[
        "A_N00R1_MIN_DELTA"
    ] = "0.0001"

    # strict resume / reproducibility
    env["nnUNet_n_proc_DA"] = "0"
    env["nnUNet_compile"] = "false"

    result_root = Path(
        env["nnUNet_results"]
    )

    fold_dir = (
        result_root
        / "Dataset501_TopAneuMRA_Location"
        / (
            "nnUNetTrainerAneurysmSparse53Retrain"
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

    complete = (
        fold_dir
        / "final_validation"
        / "COMPLETE.json"
    )

    if complete.is_file():

        print("=" * 80)
        print("A-N00R1 이미 완료")
        print("=" * 80)
        print(
            "COMPLETE:",
            complete,
        )

        return 0

    resume = (
        checkpoint_latest.is_file()
        or checkpoint_final.is_file()
    )

    command = [
        str(train_bin),
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
    print("A-N00R1 CORRECTED RETRAIN")
    print("=" * 80)

    print(
        "Mode                     :",
        "RESUME"
        if resume
        else "NEW",
    )

    print(
        "Architecture             : Standard PlainConvUNet"
    )

    print(
        "Outputs                  : 53"
    )

    print(
        "Train / Val              : 215 / 46"
    )

    print(
        "Max epochs               : 300"
    )

    print(
        "Train iterations         : 250"
    )

    print(
        "Online val iterations    : 50"
    )

    print(
        "Loss                     : GT-present-only Dice + CE"
    )

    print(
        "FG sampling              : probabilistic 0.75"
    )

    print(
        "EarlyStopping start      : 80"
    )

    print(
        "Patience                 : 40"
    )

    print(
        "Final Validation         : 46 patients ONCE"
    )

    print(
        "Strict Resume            : YES"
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
