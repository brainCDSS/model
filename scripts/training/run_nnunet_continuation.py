#!/usr/bin/env python3
import argparse
from pathlib import Path
import yaml

from brainon_ai.nnunet.config import load_config
from brainon_ai.nnunet.runner import (
    expected_output_folder,
    run_continuation_from_config,
    run_from_config,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    continuation = config["continuation"]
    experiment = config["experiment"]
    dataset = f"Dataset{int(experiment['dataset_id']):03d}_{experiment['dataset_name']}"
    parent_folder = (
        args.results_root.resolve() / dataset /
        f"nnUNetTrainerICHBaseline__{continuation['parent_output_identifier']}__{experiment['configuration']}" /
        f"fold_{experiment['fold']}"
    )
    parent_checkpoint = parent_folder / continuation["parent_checkpoint"]
    output_folder = expected_output_folder(config, args.results_root)
    latest = output_folder / "checkpoint_latest.pth"
    mode = "resume-continuation" if args.resume and latest.is_file() else "initialize-from-parent"
    print(f"mode={mode}")
    print(f"parent={parent_checkpoint}")
    print(f"output={output_folder}")
    if args.dry_run:
        if not parent_checkpoint.is_file():
            raise FileNotFoundError(parent_checkpoint)
        return 0
    if mode == "resume-continuation":
        cfg = yaml.safe_load(args.config.read_text())
        cfg["experiment"]["continue_training"] = True
        resolved = output_folder / "resume_config.yaml"
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        run_from_config(resolved, args.data_root, args.results_root, run_validation=False)
    else:
        run_continuation_from_config(
            args.config, parent_checkpoint, args.data_root, args.results_root
        )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
