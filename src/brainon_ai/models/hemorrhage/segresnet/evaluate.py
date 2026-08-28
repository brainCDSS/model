from __future__ import annotations

import argparse
from pathlib import Path

import torch

from segresnet_baseline.checkpointing import config_sha256, load_checkpoint, load_model_state, sha256_file
from segresnet_baseline.config import load_config, resolve_model_path
from segresnet_baseline.data import build_split_loader, load_manifest
from segresnet_baseline.inference import evaluate_loader, save_evaluation
from segresnet_baseline.network import build_model
from segresnet_baseline.reproducibility import seed_everything


def _resolve_checkpoint(value: str, checkpoint_dir: Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or candidate.parent != Path("."):
        return candidate.resolve()
    return checkpoint_dir / candidate.name


def main() -> None:
    parser = argparse.ArgumentParser(description="SegResNet 전체 볼륨 평가")
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--checkpoint", default="best_composite.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-save-predictions", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = resolve_model_path(config["experiment"]["output_dir"])
    manifest_path = resolve_model_path(config["experiment"]["manifest_path"])
    checkpoint_path = _resolve_checkpoint(args.checkpoint, run_dir / "checkpoints")
    requested_device = str(args.device or config["experiment"].get("device", "cuda"))
    if requested_device.lower() == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 요청했지만 torch.cuda.is_available()이 False입니다.")
    device = torch.device(requested_device)
    seed_everything(int(config["experiment"]["seed"]))

    data_root_value = config["data"].get("root_dir")
    data_root = resolve_model_path(data_root_value) if data_root_value else None
    manifest = load_manifest(manifest_path, data_root=data_root)
    checkpoint = load_checkpoint(checkpoint_path, device)
    if checkpoint.get("manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("checkpoint와 현재 manifest의 SHA-256이 다릅니다.")
    if checkpoint.get("config_sha256") != config_sha256(config):
        raise RuntimeError("checkpoint와 현재 YAML 설정이 다릅니다. 학습 때 저장된 config_resolved.yaml을 사용하세요.")
    model = build_model(config).to(device)
    load_model_state(model, checkpoint)
    loader = build_split_loader(config, manifest, args.split)
    output_dir = run_dir / "evaluation" / args.split / checkpoint_path.stem
    prediction_dir = None
    if bool(config["inference"]["save_predictions"]) and not args.no_save_predictions:
        prediction_dir = output_dir / "predictions"
    summary, per_case = evaluate_loader(
        model,
        loader,
        config,
        device,
        prediction_dir=prediction_dir,
        description=args.split,
    )
    summary.update(
        {
            "split": args.split,
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(checkpoint["epoch"]) + 1,
        }
    )
    save_evaluation(summary, per_case, output_dir)
    print(f"Dice={summary['dice_positive_macro']:.4f}")
    print(f"Recall={summary['recall_positive_macro']:.4f}")
    print(f"Composite={summary['composite']:.4f}")
    print(f"저장: {output_dir}")


if __name__ == "__main__":
    main()
