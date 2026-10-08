from __future__ import annotations

import argparse

from brainon_ai.models.aneurysm.segresnet.config import (
    load_config,
    resolve_repository_path,
)
from brainon_ai.models.aneurysm.segresnet.evaluate import (
    final_validation,
)
from brainon_ai.models.aneurysm.segresnet.train import (
    train,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "A-S00 Standard SegResNet training "
            "and final validation"
        )
    )

    parser.add_argument(
        "--config",
        default=None,
        help="A-S00 YAML path",
    )

    parser.add_argument(
        "--stage",
        choices=("train", "final", "all"),
        default="train",
        help=(
            "train: patch training only; "
            "final: final 46-patient validation only; "
            "all: train then final"
        ),
    )

    arguments = parser.parse_args()
    config = load_config(arguments.config)

    run_dir = resolve_repository_path(
        config["experiment"]["output_dir"]
    )

    if arguments.stage in {"train", "all"}:
        run_dir = train(config)

        final_checkpoint = (
            run_dir
            / "checkpoints"
            / "checkpoint_final.pth"
        )

        if not final_checkpoint.is_file():
            raise RuntimeError(
                "Training ended without "
                "checkpoint_final.pth"
            )

        print(
            "PASS: A-S00 patch training stage complete"
        )
        print(
            "Final validation has not run:",
            not (
                run_dir
                / "final_validation"
                / "COMPLETE.json"
            ).exists(),
        )

    if arguments.stage in {"final", "all"}:
        final_dir = final_validation(
            config,
            run_dir,
        )

        complete_path = (
            final_dir / "COMPLETE.json"
        )

        if not complete_path.is_file():
            raise RuntimeError(
                "A-S00 final validation ended "
                "without COMPLETE.json"
            )

        print(
            "PASS: A-S00 final validation complete"
        )
        print(
            "Patient CSV:",
            final_dir
            / "per_patient_metrics.csv",
        )
        print(
            "Class CSV  :",
            final_dir
            / "per_class_metrics.csv",
        )
        print(
            "Summary    :",
            final_dir / "summary.json",
        )
        print(
            "Completion :",
            complete_path,
        )


if __name__ == "__main__":
    main()
