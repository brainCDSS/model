#!/usr/bin/env python3

"""
A-N00R1 Final Validation 완료 여부 및
46-patient 결과 파일 확인.

실제 inference/metric 계산은 Trainer에서 수행한다.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPECTED_CASES = 46


def main() -> int:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold-dir",
        required=True,
    )

    args = parser.parse_args()

    fold_dir = Path(
        args.fold_dir
    ).resolve()

    final_dir = (
        fold_dir
        / "final_validation"
    )

    complete_file = (
        final_dir
        / "COMPLETE.json"
    )

    patient_file = (
        final_dir
        / "per_patient_metrics.csv"
    )

    print("=" * 80)
    print("A-N00R1 FINAL VALIDATION 확인")
    print("=" * 80)

    print(
        "Final directory :",
        final_dir,
    )

    print(
        "COMPLETE.json   :",
        complete_file.is_file(),
    )

    print(
        "Patient metrics :",
        patient_file.is_file(),
    )

    if not complete_file.is_file():

        print(
            "STATUS          : NOT COMPLETE"
        )

        return 2

    if not patient_file.is_file():

        raise RuntimeError(
            "COMPLETE.json은 존재하지만 "
            "per_patient_metrics.csv가 없습니다."
        )

    with patient_file.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    print(
        "Patient rows    :",
        len(rows),
    )

    if len(rows) != EXPECTED_CASES:

        raise RuntimeError(
            "Final Validation 환자 수 오류: "
            f"{len(rows)}/{EXPECTED_CASES}"
        )

    complete_data = json.loads(
        complete_file.read_text(
            encoding="utf-8"
        )
    )

    print(
        "STATUS          : COMPLETE"
    )

    print(
        "Metadata keys   :",
        sorted(
            complete_data.keys()
        ),
    )

    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
