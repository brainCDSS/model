from __future__ import annotations

import argparse

import torch

from segresnet_baseline.config import load_config, resolve_model_path
from segresnet_baseline.data import build_loaders, build_train_transform, load_manifest, records_for_split
from segresnet_baseline.losses import build_loss
from segresnet_baseline.metrics import binary_case_metrics
from segresnet_baseline.network import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="SegResNet 설정/manifest/forward 점검")
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--with-data-sample", action="store_true")
    parser.add_argument("--with-loader", action="store_true", help="설정된 worker/collate로 train batch 하나 점검")
    parser.add_argument("--full-patch", action="store_true", help="설정된 실제 patch/batch로 backward까지 점검")
    args = parser.parse_args()

    config = load_config(args.config)
    manifest_path = resolve_model_path(config["experiment"]["manifest_path"])
    data_root_value = config["data"].get("root_dir")
    data_root = resolve_model_path(data_root_value) if data_root_value else None
    manifest = load_manifest(manifest_path, data_root=data_root)
    counts = manifest.groupby("split").size().to_dict()
    if counts != {"test": 40, "train": 186, "validation": 40}:
        raise RuntimeError(f"예상과 다른 split 개수: {counts}")

    device = torch.device(args.device)
    model = build_model(config).to(device)
    image = torch.randn(1, 1, 32, 32, 16, device=device)
    target = torch.zeros(1, 1, 32, 32, 16, dtype=torch.long, device=device)
    target[:, :, 14:18, 14:18, 7:9] = 1
    logits = model(image)
    if logits.shape != (1, 2, 32, 32, 16):
        raise RuntimeError(f"예상과 다른 model output: {tuple(logits.shape)}")
    loss = build_loss(config)(logits, target)
    loss.backward()
    if not torch.isfinite(loss):
        raise RuntimeError("loss가 유한하지 않습니다.")

    metric = binary_case_metrics(torch.tensor([1, 0]), torch.tensor([1, 1]))
    if abs(float(metric["dice"]) - 2 / 3) > 1e-6 or abs(float(metric["recall"]) - 0.5) > 1e-6:
        raise RuntimeError(f"metric sanity check 실패: {metric}")

    if args.with_data_sample:
        records = records_for_split(manifest, "train")
        positive_record = next(record for record in records if record["positive_voxels"] > 0)
        empty_record = next(record for record in records if record["positive_voxels"] == 0)
        transform = build_train_transform(config)
        expected_shape = (1, *tuple(int(x) for x in config["data"]["patch_size"]))
        for kind, record in (("positive", positive_record), ("empty", empty_record)):
            patches = transform(record)
            if not isinstance(patches, list) or len(patches) != int(config["data"]["patches_per_volume"]):
                raise RuntimeError(f"{kind} patch sampler의 반환 개수가 예상과 다릅니다.")
            for patch in patches:
                if tuple(patch["image"].shape) != expected_shape or tuple(patch["label"].shape) != expected_shape:
                    raise RuntimeError(
                        f"{kind} patch shape이 예상과 다릅니다: image={tuple(patch['image'].shape)}, "
                        f"label={tuple(patch['label'].shape)}"
                    )
                labels = set(torch.unique(patch["label"]).tolist())
                if not labels.issubset({0, 1}):
                    raise RuntimeError(f"{kind} patch의 유효하지 않은 label 값: {labels}")
                if kind == "empty" and int(torch.count_nonzero(patch["label"])) != 0:
                    raise RuntimeError("빈 GT 사례에서 양성 label이 나타났습니다.")

    if args.full_patch:
        effective_batch = int(config["data"]["batch_size"]) * int(config["data"]["patches_per_volume"])
        patch_size = tuple(int(x) for x in config["data"]["patch_size"])
        model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        full_image = torch.randn(effective_batch, 1, *patch_size, device=device)
        full_target = torch.zeros(effective_batch, 1, *patch_size, dtype=torch.long, device=device)
        centers = tuple(size // 2 for size in patch_size)
        full_target[
            :,
            :,
            centers[0] - 2 : centers[0] + 2,
            centers[1] - 2 : centers[1] + 2,
            centers[2] - 1 : centers[2] + 1,
        ] = 1
        amp_enabled = device.type == "cuda" and bool(config["training"]["amp"])
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            full_loss = build_loss(config)(model(full_image), full_target)
        full_loss.backward()
        message = f"full patch PASS: effective_batch={effective_batch}, patch={patch_size}"
        if device.type == "cuda":
            peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
            message += f", peak_allocated={peak_gib:.2f} GiB"
        print(message)

    if args.with_loader:
        batch = next(iter(build_loaders(config, manifest)["train"]))
        expected_batch = int(config["data"]["batch_size"]) * int(config["data"]["patches_per_volume"])
        expected_shape = (expected_batch, 1, *tuple(int(x) for x in config["data"]["patch_size"]))
        if tuple(batch["image"].shape) != expected_shape or tuple(batch["label"].shape) != expected_shape:
            raise RuntimeError(
                f"loader batch shape이 예상과 다릅니다: image={tuple(batch['image'].shape)}, "
                f"label={tuple(batch['label'].shape)}, expected={expected_shape}"
            )
        print(f"loader PASS: image={tuple(batch['image'].shape)}, label={tuple(batch['label'].shape)}")

    print(f"PASS: manifest={counts}, output={tuple(logits.shape)}, loss={float(loss.detach()):.4f}")
    if not args.with_data_sample:
        print("실제 NIfTI patch까지 확인하려면 --with-data-sample을 추가하세요.")


if __name__ == "__main__":
    main()
