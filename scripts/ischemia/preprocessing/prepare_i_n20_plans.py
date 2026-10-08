#!/usr/bin/env python3
"""Create I-N20 ResidualEncoderUNet plans while reusing nnUNet preprocessed data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    experiment, plans = config["experiment"], config["plans"]
    dataset = f"Dataset{int(experiment['dataset_id']):03d}_{experiment['dataset_name']}"
    folder = args.data_root.resolve() / "nnUNet_preprocessed" / dataset
    source = folder / f"{plans['base_identifier']}.json"
    destination = folder / f"{plans['output_identifier']}.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    architecture = payload["configurations"][experiment["configuration"]]["architecture"]
    kwargs = architecture["arch_kwargs"]
    stages = int(kwargs["n_stages"])
    kwargs.pop("n_conv_per_stage", None)
    kwargs["n_blocks_per_stage"] = [1, 3, 4, 6, 6][:stages]
    kwargs["n_conv_per_stage_decoder"] = [1] * (stages - 1)
    architecture["network_class_name"] = (
        "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet"
    )
    payload["plans_name"] = plans["output_identifier"]
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
