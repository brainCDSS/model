from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from brainon_ai.models.aneurysm.segresnet.checkpointing import (
    atomic_json_save,
    atomic_torch_save,
    build_checkpoint,
    load_checkpoint,
    load_training_state,
    sha256_file,
    validate_checkpoint,
)
from brainon_ai.models.aneurysm.segresnet.config import (
    resolve_repository_path,
    save_resolved_config,
)
from brainon_ai.models.aneurysm.segresnet.data import (
    build_patch_datasets,
    deterministic_batch,
    deterministic_indices,
    load_manifest,
)
from brainon_ai.models.aneurysm.segresnet.losses import (
    build_loss,
)
from brainon_ai.models.aneurysm.segresnet.metrics import (
    confusion_by_class,
    foreground_dice,
)
from brainon_ai.models.aneurysm.segresnet.network import (
    build_model,
)
from brainon_ai.models.aneurysm.segresnet.reproducibility import (
    seed_everything,
)


def _device(config: dict) -> torch.device:
    requested = str(
        config["experiment"]["device"]
    )

    if (
        requested.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "A-S00 requires CUDA, but "
            "torch.cuda.is_available() is False"
        )

    return torch.device(requested)


def _grad_scaler(
    device: torch.device,
    enabled: bool,
):
    try:
        return torch.amp.GradScaler(
            device.type,
            enabled=enabled,
        )
    except TypeError:
        return torch.cuda.amp.GradScaler(
            enabled=enabled
        )


def _optimizer(
    model: torch.nn.Module,
    config: dict,
) -> torch.optim.Optimizer:
    settings = config["optimizer"]

    if str(settings["name"]).lower() != "adamw":
        raise ValueError(
            "Unsupported A-S00 optimizer: "
            f"{settings['name']}"
        )

    return torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(
            settings["weight_decay"]
        ),
    )


def _scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict,
):
    settings = config["scheduler"]

    if (
        str(settings["name"]).lower()
        != "polynomial"
    ):
        raise ValueError(
            "Unsupported A-S00 scheduler: "
            f"{settings['name']}"
        )

    max_epochs = int(
        config["training"]["max_epochs"]
    )
    initial_lr = float(
        config["optimizer"]["learning_rate"]
    )
    minimum_lr = float(
        settings["min_learning_rate"]
    )
    power = float(settings["power"])

    def factor(epoch: int) -> float:
        fraction = max(
            0.0,
            1.0
            - min(epoch, max_epochs)
            / max_epochs,
        )

        learning_rate = (
            minimum_lr
            + (initial_lr - minimum_lr)
            * fraction**power
        )

        return learning_rate / initial_lr

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=factor,
    )


def _new_progress(
    num_classes: int,
) -> dict:
    return {
        "phase": "train",
        "epoch": 0,
        "completed_train_iterations": 0,
        "completed_validation_iterations": 0,
        "partial_train_losses": [],
        "partial_validation_losses": [],
        "validation_tp": [0] * num_classes,
        "validation_fp": [0] * num_classes,
        "validation_fn": [0] * num_classes,
        "ema_fg_dice": None,
        "best_ema_fg_dice": None,
        "early_stopping_best": None,
        "early_stopping_bad_epochs": 0,
        "history": [],
        "global_optimizer_steps": 0,
        "stopped_early": False,
    }


def _json_number(value):
    if value is None:
        return None

    number = float(value)

    return (
        number
        if math.isfinite(number)
        else None
    )


def _write_runtime_files(
    run_dir: Path,
    progress: dict,
    config: dict,
) -> None:
    history = pd.DataFrame(
        progress["history"]
    )

    history_path = (
        run_dir / "history.csv"
    )
    temporary = history_path.with_name(
        history_path.name + ".tmp"
    )

    history.to_csv(
        temporary,
        index=False,
        encoding="utf-8-sig",
    )

    os.replace(
        temporary,
        history_path,
    )

    atomic_json_save(
        {
            "phase": progress["phase"],
            "epoch": int(progress["epoch"]),
            "max_epochs": int(
                config["training"]["max_epochs"]
            ),
            "completed_train_iterations": int(
                progress[
                    "completed_train_iterations"
                ]
            ),
            "train_iterations_per_epoch": int(
                config["training"][
                    "iterations_per_epoch"
                ]
            ),
            "completed_validation_iterations": int(
                progress[
                    "completed_validation_iterations"
                ]
            ),
            "validation_iterations_per_epoch": int(
                config["training"][
                    "validation_iterations_per_epoch"
                ]
            ),
            "global_optimizer_steps": int(
                progress["global_optimizer_steps"]
            ),
            "ema_fg_dice": _json_number(
                progress["ema_fg_dice"]
            ),
            "best_ema_fg_dice": _json_number(
                progress["best_ema_fg_dice"]
            ),
            "early_stopping_bad_epochs": int(
                progress[
                    "early_stopping_bad_epochs"
                ]
            ),
            "updated_at": datetime.now(
                timezone.utc
            ).isoformat(),
        },
        run_dir / "progress.json",
    )


def _save_latest(
    run_dir: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    config,
    manifest_hash,
    progress,
) -> dict:
    payload = build_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        config=config,
        manifest_hash=manifest_hash,
        progress=progress,
    )

    atomic_torch_save(
        payload,
        run_dir
        / "checkpoints"
        / "checkpoint_latest.pth",
    )

    _write_runtime_files(
        run_dir,
        progress,
        config,
    )

    return payload


def _print_policy(
    config: dict,
    mode: str,
    run_dir: Path,
) -> None:
    print("[A-S00 TRAINING]")
    print("Mode                    :", mode)
    print(
        "Model                   :",
        "Standard MONAI SegResNet",
    )
    print(
        "Target                  :",
        "Location, 53 classes",
    )
    print(
        "Train / Validation      :",
        "215 / 46",
    )
    print("Test                    : excluded")
    print(
        "Patch [X,Y,Z]           :",
        config["data"]["patch_size"],
    )
    print(
        "Batch                   :",
        config["data"]["batch_size"],
    )
    print(
        "Train iterations        :",
        config["training"][
            "iterations_per_epoch"
        ],
        "/ epoch",
    )
    print(
        "Online patch validation :",
        config["training"][
            "validation_iterations_per_epoch"
        ],
        "/ epoch",
    )
    print(
        "Max epochs              :",
        config["training"]["max_epochs"],
    )
    print(
        "Best / EarlyStopping    :",
        "ema_fg_dice",
    )
    print(
        "Full-volume per epoch   : NO"
    )
    print(
        "Final full-volume       :",
        "best checkpoint, 46 patients, once",
    )
    print(
        "Strict resume           :",
        "next unfinished unit",
    )
    print(
        "Run directory           :",
        run_dir,
    )


def train(config: dict) -> Path:
    device = _device(config)

    seed = int(
        config["experiment"]["seed"]
    )
    seed_everything(seed)

    manifest = load_manifest(config)

    manifest_path = resolve_repository_path(
        config["data"]["manifest_path"]
    )
    manifest_hash = sha256_file(
        manifest_path
    )

    run_dir = resolve_repository_path(
        config["experiment"]["output_dir"]
    )

    checkpoint_dir = (
        run_dir / "checkpoints"
    )
    latest_path = (
        checkpoint_dir
        / "checkpoint_latest.pth"
    )
    best_path = (
        checkpoint_dir
        / "checkpoint_best.pth"
    )
    final_path = (
        checkpoint_dir
        / "checkpoint_final.pth"
    )
    complete_path = (
        run_dir
        / str(config["completion"]["marker"])
    )

    if complete_path.is_file():
        print(
            "A-S00 ALREADY COMPLETE:",
            complete_path,
        )
        return run_dir

    existing_files = (
        list(run_dir.rglob("*"))
        if run_dir.exists()
        else []
    )

    if (
        existing_files
        and not latest_path.is_file()
        and not final_path.is_file()
    ):
        raise RuntimeError(
            "A-S00 run directory is non-empty "
            "but has no resume checkpoint: "
            f"{run_dir}"
        )

    datasets = build_patch_datasets(
        config,
        manifest,
    )

    model = build_model(config).to(device)
    loss_function = build_loss(config)
    optimizer = _optimizer(model, config)
    scheduler = _scheduler(
        optimizer,
        config,
    )

    amp_enabled = bool(
        config["training"]["amp"]
        and device.type == "cuda"
    )

    scaler = _grad_scaler(
        device,
        amp_enabled,
    )

    if latest_path.is_file():
        checkpoint = load_checkpoint(
            latest_path,
            device,
        )
        validate_checkpoint(
            checkpoint,
            config,
            manifest_hash,
        )
        progress = load_training_state(
            model,
            optimizer,
            scheduler,
            scaler,
            checkpoint,
        )
        mode = "RESUME"

    elif final_path.is_file():
        checkpoint = load_checkpoint(
            final_path,
            device,
        )
        validate_checkpoint(
            checkpoint,
            config,
            manifest_hash,
        )
        progress = load_training_state(
            model,
            optimizer,
            scheduler,
            scaler,
            checkpoint,
        )
        mode = "FINAL-VALIDATION-RESUME"

    else:
        progress = _new_progress(
            int(config["model"]["out_channels"])
        )
        mode = "NEW"

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_resolved_config(
        config,
        run_dir / "config_resolved.yaml",
    )

    manifest.to_csv(
        run_dir / "manifest_snapshot.csv",
        index=False,
        encoding="utf-8-sig",
    )

    _print_policy(
        config,
        mode,
        run_dir,
    )

    train_iterations = int(
        config["training"][
            "iterations_per_epoch"
        ]
    )
    validation_iterations = int(
        config["training"][
            "validation_iterations_per_epoch"
        ]
    )
    max_epochs = int(
        config["training"]["max_epochs"]
    )
    batch_size = int(
        config["data"]["batch_size"]
    )
    progress_every = int(
        config["training"][
            "progress_every_iterations"
        ]
    )
    clip_norm = float(
        config["training"]["gradient_clip_norm"]
    )
    num_classes = int(
        config["model"]["out_channels"]
    )

    while progress["phase"] not in {
        "training_complete",
        "final_validation",
        "complete",
    }:
        epoch = int(progress["epoch"])

        if epoch >= max_epochs:
            progress["phase"] = (
                "training_complete"
            )
            break

        if progress["phase"] == "train":
            completed = int(
                progress[
                    "completed_train_iterations"
                ]
            )

            while completed < train_iterations:
                indices = deterministic_indices(
                    case_count=len(
                        datasets["train"]
                    ),
                    batch_size=batch_size,
                    seed=seed,
                    epoch=epoch,
                    iteration=completed,
                    phase="train",
                )

                batch = deterministic_batch(
                    datasets["train"],
                    indices=indices,
                    transform_seed=(
                        seed
                        + epoch * 1_000_003
                        + completed
                    ),
                )

                image = batch["image"].to(
                    device,
                    non_blocking=False,
                )
                label = batch["label"].to(
                    device,
                    non_blocking=False,
                )

                model.train()
                optimizer.zero_grad(
                    set_to_none=True
                )

                with torch.autocast(
                    device_type=device.type,
                    enabled=amp_enabled,
                ):
                    logits = model(image)
                    loss = loss_function(
                        logits,
                        label,
                    )

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "Non-finite train loss at "
                        f"epoch={epoch + 1}, "
                        f"unit={completed + 1}"
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                if clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        clip_norm,
                    )

                scaler.step(optimizer)
                scaler.update()

                progress[
                    "partial_train_losses"
                ].append(
                    float(loss.detach().cpu())
                )

                completed += 1
                progress[
                    "completed_train_iterations"
                ] = completed
                progress[
                    "global_optimizer_steps"
                ] += 1

                _save_latest(
                    run_dir,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    config,
                    manifest_hash,
                    progress,
                )

                if (
                    completed % progress_every == 0
                    or completed == train_iterations
                ):
                    print(
                        "[TRAIN] "
                        f"epoch {epoch + 1} | "
                        f"{completed}/{train_iterations}",
                        flush=True,
                    )

            progress["phase"] = (
                "online_validation"
            )
            progress[
                "completed_validation_iterations"
            ] = 0
            progress[
                "partial_validation_losses"
            ] = []
            progress["validation_tp"] = (
                [0] * num_classes
            )
            progress["validation_fp"] = (
                [0] * num_classes
            )
            progress["validation_fn"] = (
                [0] * num_classes
            )

            _save_latest(
                run_dir,
                model,
                optimizer,
                scheduler,
                scaler,
                config,
                manifest_hash,
                progress,
            )

        if progress["phase"] == (
            "online_validation"
        ):
            completed = int(
                progress[
                    "completed_validation_iterations"
                ]
            )

            model.eval()

            while completed < validation_iterations:
                # 동일한 고정 online-validation patch
                # 집합으로 epoch 간 비교합니다.
                indices = deterministic_indices(
                    case_count=len(
                        datasets["validation"]
                    ),
                    batch_size=batch_size,
                    seed=seed,
                    epoch=0,
                    iteration=completed,
                    phase="online_validation",
                )

                batch = deterministic_batch(
                    datasets["validation"],
                    indices=indices,
                    transform_seed=(
                        seed
                        + 2_000_000_011
                        + completed
                    ),
                )

                image = batch["image"].to(
                    device,
                    non_blocking=False,
                )
                label = batch["label"].to(
                    device,
                    non_blocking=False,
                )

                with (
                    torch.inference_mode(),
                    torch.autocast(
                        device_type=device.type,
                        enabled=amp_enabled,
                    ),
                ):
                    logits = model(image)
                    loss = loss_function(
                        logits,
                        label,
                    )

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "Non-finite validation loss at "
                        f"epoch={epoch + 1}, "
                        f"unit={completed + 1}"
                    )

                prediction = torch.argmax(
                    logits,
                    dim=1,
                )

                tp, fp, fn = confusion_by_class(
                    prediction,
                    label[:, 0],
                    num_classes,
                )

                progress[
                    "partial_validation_losses"
                ].append(
                    float(loss.detach().cpu())
                )

                progress["validation_tp"] = (
                    np.asarray(
                        progress["validation_tp"],
                        dtype=np.int64,
                    )
                    + tp
                ).tolist()

                progress["validation_fp"] = (
                    np.asarray(
                        progress["validation_fp"],
                        dtype=np.int64,
                    )
                    + fp
                ).tolist()

                progress["validation_fn"] = (
                    np.asarray(
                        progress["validation_fn"],
                        dtype=np.int64,
                    )
                    + fn
                ).tolist()

                completed += 1
                progress[
                    "completed_validation_iterations"
                ] = completed

                _save_latest(
                    run_dir,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    config,
                    manifest_hash,
                    progress,
                )

                if (
                    completed % progress_every == 0
                    or completed
                    == validation_iterations
                ):
                    print(
                        "[ONLINE-VAL] "
                        f"epoch {epoch + 1} | "
                        f"{completed}/"
                        f"{validation_iterations}",
                        flush=True,
                    )

            tp = np.asarray(
                progress["validation_tp"],
                dtype=np.int64,
            )
            fp = np.asarray(
                progress["validation_fp"],
                dtype=np.int64,
            )
            fn = np.asarray(
                progress["validation_fn"],
                dtype=np.int64,
            )

            patch_dice = foreground_dice(
                tp,
                fp,
                fn,
            )

            if not math.isfinite(patch_dice):
                raise RuntimeError(
                    "Online validation foreground "
                    "Dice is non-finite"
                )

            previous_ema = progress[
                "ema_fg_dice"
            ]
            alpha = float(
                config["validation"][
                    "online_patch"
                ]["ema_alpha"]
            )

            ema = (
                patch_dice
                if previous_ema is None
                else (
                    alpha * float(previous_ema)
                    + (1.0 - alpha)
                    * patch_dice
                )
            )

            progress["ema_fg_dice"] = float(
                ema
            )

            best = progress[
                "best_ema_fg_dice"
            ]
            is_best = (
                best is None
                or ema > float(best)
            )

            if is_best:
                progress[
                    "best_ema_fg_dice"
                ] = float(ema)

            early_best = progress[
                "early_stopping_best"
            ]
            minimum_delta = float(
                config["early_stopping"][
                    "min_delta"
                ]
            )

            if (
                early_best is None
                or ema
                > float(early_best)
                + minimum_delta
            ):
                progress[
                    "early_stopping_best"
                ] = float(ema)
                progress[
                    "early_stopping_bad_epochs"
                ] = 0
            else:
                progress[
                    "early_stopping_bad_epochs"
                ] += 1

            progress["history"].append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(
                        np.mean(
                            progress[
                                "partial_train_losses"
                            ]
                        )
                    ),
                    "online_validation_loss": float(
                        np.mean(
                            progress[
                                "partial_validation_losses"
                            ]
                        )
                    ),
                    "online_patch_mean_fg_dice": float(
                        patch_dice
                    ),
                    "online_patch_ema_fg_dice": float(
                        ema
                    ),
                    "learning_rate": float(
                        optimizer.param_groups[
                            0
                        ]["lr"]
                    ),
                }
            )

            scheduler.step()
            progress["epoch"] = epoch + 1

            should_stop = bool(
                config["early_stopping"]["enabled"]
                and progress["epoch"]
                > int(
                    config["early_stopping"][
                        "start_epoch"
                    ]
                )
                and progress[
                    "early_stopping_bad_epochs"
                ]
                >= int(
                    config["early_stopping"][
                        "patience"
                    ]
                )
            )

            progress["stopped_early"] = (
                should_stop
            )

            progress["phase"] = (
                "training_complete"
                if (
                    should_stop
                    or progress["epoch"]
                    >= max_epochs
                )
                else "train"
            )

            progress[
                "completed_train_iterations"
            ] = 0
            progress[
                "completed_validation_iterations"
            ] = 0
            progress[
                "partial_train_losses"
            ] = []
            progress[
                "partial_validation_losses"
            ] = []

            payload = build_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                manifest_hash=manifest_hash,
                progress=progress,
            )

            if is_best:
                atomic_torch_save(
                    payload,
                    best_path,
                )

            atomic_torch_save(
                payload,
                latest_path,
            )

            _write_runtime_files(
                run_dir,
                progress,
                config,
            )

            print(
                f"[EPOCH {epoch + 1}] "
                f"patch_fg_dice={patch_dice:.6f} "
                f"ema_fg_dice={ema:.6f} "
                "bad_epochs="
                f"{progress['early_stopping_bad_epochs']}/"
                f"{config['early_stopping']['patience']}",
                flush=True,
            )

    if progress["phase"] == (
        "training_complete"
    ):
        progress["phase"] = (
            "final_validation"
        )

        payload = _save_latest(
            run_dir,
            model,
            optimizer,
            scheduler,
            scaler,
            config,
            manifest_hash,
            progress,
        )

        atomic_torch_save(
            payload,
            final_path,
        )

    elif (
        progress["phase"] == "final_validation"
        and not final_path.is_file()
    ):
        payload = build_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            manifest_hash=manifest_hash,
            progress=progress,
        )

        atomic_torch_save(
            payload,
            final_path,
        )

    if not best_path.is_file():
        raise RuntimeError(
            "Training ended without "
            "checkpoint_best.pth"
        )

    return run_dir
