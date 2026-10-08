from __future__ import annotations

import importlib
import json
import math
import os
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from multiprocessing.pool import ThreadPool
from pathlib import Path
from time import monotonic

import numpy as np
import torch
import torch.distributed as dist
from torch._dynamo import OptimizedModule

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import (
    NonDetMultiThreadedAugmenter, SingleThreadedAugmenter, get_allowed_n_proc_DA,
    infer_dataset_class, nnUNetDataLoader,
)
from nnunetv2.utilities.collate_outputs import collate_outputs

from brainon_ai.nnunet.config import config_digest, load_config
from brainon_ai.nnunet.losses import build_loss
from brainon_ai.nnunet.networks import build_network
from brainon_ai.nnunet.pruning import EarlyStopping
from brainon_ai.nnunet.sampling.small_lesion import build_sampling_probabilities, write_sampling_report
from brainon_ai.nnunet.sampling.component_patch import (
    ComponentCenteredDataLoader, build_component_centers, write_component_report,
)


class nnUNetTrainerICHBaseline(nnUNetTrainer):
    """YAML로 trainer/loss/network/checkpoint/pruning을 제어하는 단일 GPU baseline."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        self.ich_config = load_config()
        self.ich_config_digest = config_digest(self.ich_config)
        plans = deepcopy(plans)
        plan_config = plans.get("configurations", {}).get(configuration)
        if plan_config is None:
            raise ValueError(f"plans에 configuration이 없습니다: {configuration}")
        plan_config["patch_size"] = [
            int(value) for value in self.ich_config["plans"]["patch_size"]
        ]
        plan_config["batch_size"] = int(self.ich_config["plans"]["batch_size"])
        super().__init__(plans, configuration, fold, dataset_json, device)
        cfg = self.ich_config["trainer"]
        self.num_epochs = int(cfg["epochs"])
        self.num_iterations_per_epoch = int(cfg["iterations_per_epoch"])
        self.num_val_iterations_per_epoch = int(cfg["validation_iterations_per_epoch"])
        self.save_every = int(cfg["save_every"])
        self.initial_lr = float(cfg["learning_rate"])
        self.weight_decay = float(cfg["weight_decay"])
        self.oversample_foreground_percent = float(cfg["oversample_foreground_percent"])

        pruning = self.ich_config["pruning"]
        self.early_stopping = EarlyStopping(
            enabled=bool(pruning["enabled"]),
            min_epochs=int(pruning["min_epochs"]),
            patience=int(pruning["patience"]),
            min_delta=float(pruning["min_delta"]),
        )
        self._last_dice = float("nan")
        self._last_recall = float("nan")
        self._last_composite = float("nan")
        self._stop_requested = False
        self._best_dice = float("-inf")
        self._best_recall = float("-inf")
        self._best_composite = float("-inf")
        self._metrics_path = Path(self.output_folder) / "custom_metrics.json"
        self._progress_path = Path(self.output_folder) / "progress.json"
        self._config_marker_path = Path(self.output_folder) / "ich_config.json"
        self._session_started = monotonic()
        self._epoch_durations: list[float] = []
        full_validation = self.ich_config.get("full_validation", {})
        self._full_validation_enabled = bool(full_validation.get("enabled", False))
        self._full_validation_interval = int(full_validation.get("interval", 5))
        self._full_validation_start_epoch = int(
            full_validation.get("start_epoch", self._full_validation_interval)
        )
        if self._full_validation_interval < 1 or self._full_validation_start_epoch < 1:
            raise ValueError("full_validation interval/start_epoch must be positive")
        self._archive_every_epoch = bool(full_validation.get("archive_every_epoch", False))
        self._full_validation_top_k = int(full_validation.get("keep_top_k", 3))
        self._full_validation_neighborhood_radius = int(
            full_validation.get("neighborhood_radius", 4)
        )
        self._best_full_macro_dice = float("-inf")
        self._best_full_macro_epoch = None
        self._last_full_macro_dice = float("nan")
        self._full_validation_history_path = Path(self.output_folder) / "full_validation_history.json"

    @staticmethod
    def _finite_or_none(value: float):
        return float(value) if math.isfinite(value) else None

    def _write_json_atomic(self, path: Path, payload: dict) -> None:
        if self.local_rank != 0:
            return
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)

    def _write_progress(
        self,
        status: str,
        completed_epochs: int,
        dice: float = float("nan"),
        recall: float = float("nan"),
        composite: float = float("nan"),
        message: str | None = None,
    ) -> None:
        if not math.isfinite(dice):
            dice = self._last_dice
        if not math.isfinite(recall):
            recall = self._last_recall
        if not math.isfinite(composite):
            composite = self._last_composite
        mean_epoch_seconds = (
            sum(self._epoch_durations) / len(self._epoch_durations)
            if self._epoch_durations
            else None
        )
        payload = {
            "status": status,
            "message": message,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "current_epoch": completed_epochs,
            "total_epochs": self.num_epochs,
            "progress_percent": round(100 * completed_epochs / self.num_epochs, 2),
            "session_elapsed_seconds": round(monotonic() - self._session_started, 2),
            "last_epoch_seconds": self._epoch_durations[-1] if self._epoch_durations else None,
            "mean_epoch_seconds": mean_epoch_seconds,
            "eta_seconds": (
                mean_epoch_seconds * max(0, self.num_epochs - completed_epochs)
                if mean_epoch_seconds
                else None
            ),
            "train_loss": (
                float(self.logger.get_value("train_losses", step=-1))
                if completed_epochs > 0
                else None
            ),
            "pseudo_dice": self._finite_or_none(dice),
            "pseudo_recall": self._finite_or_none(recall),
            "composite": self._finite_or_none(composite),
            "best_dice": self._finite_or_none(self._best_dice),
            "best_recall": self._finite_or_none(self._best_recall),
            "best_composite": self._finite_or_none(self._best_composite),
            "full_macro_dice": self._finite_or_none(self._last_full_macro_dice),
            "best_full_macro_dice": self._finite_or_none(self._best_full_macro_dice),
            "best_full_macro_epoch": self._best_full_macro_epoch,
            "top_full_macro_epochs": self._top_full_macro_epochs(),
            "retained_epoch_checkpoints": self._retained_checkpoint_count(),
            "pruning_bad_checks": self.early_stopping.bad_checks,
            "pruning_patience": self.early_stopping.patience,
            "output_folder": str(self.output_folder),
            "batch_size": self.configuration_manager.batch_size,
            "patch_size": list(self.configuration_manager.patch_size),
            "gpu_max_memory_gb": (
                round(torch.cuda.max_memory_allocated(self.device) / (1024 ** 3), 3)
                if self.device.type == "cuda"
                else None
            ),
        }
        self._write_json_atomic(self._progress_path, payload)

    def on_train_start(self):
        if self._config_marker_path.is_file():
            marker = json.loads(self._config_marker_path.read_text(encoding="utf-8"))
            if marker.get("sha256") != self.ich_config_digest:
                raise RuntimeError(
                    "같은 nnU-Net 결과 폴더에 다른 YAML 설정을 사용할 수 없습니다. "
                    "plans.output_identifier를 새 이름으로 바꾸세요."
                )
        super().on_train_start()
        history = self._load_full_validation_history()
        validated_epochs = {int(item["epoch"]) for item in history}
        if (
            self._full_validation_enabled
            and self.current_epoch > 0
            and self.current_epoch >= self._full_validation_start_epoch
            and (
                self.current_epoch - self._full_validation_start_epoch
            ) % self._full_validation_interval == 0
            and self.current_epoch not in validated_epochs
        ):
            checkpoint = (
                Path(self.output_folder)
                / "epoch_checkpoints"
                / f"checkpoint_epoch_{self.current_epoch:04d}.pth"
            )
            self.print_to_log_file(
                f"Resuming missing full-volume validation at epoch {self.current_epoch}"
            )
            self._maybe_run_full_validation(
                self.current_epoch,
                checkpoint if checkpoint.is_file() else None,
            )
        self._write_progress("running", self.current_epoch)
        if self.local_rank == 0:
            payload = {
                "sha256": self.ich_config_digest,
                "config": {key: value for key, value in self.ich_config.items() if not key.startswith("_")},
            }
            self._write_json_atomic(self._config_marker_path, payload)

    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank != 0:
            return
        if self.disable_checkpointing:
            self.print_to_log_file("No checkpoint written, checkpointing is disabled")
            return
        module = self.network.module if self.is_ddp else self.network
        if isinstance(module, OptimizedModule):
            module = module._orig_mod
        checkpoint = {
            "network_weights": module.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "grad_scaler_state": self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
            "logging": self.logger.get_checkpoint(),
            "_best_ema": self._best_ema,
            "current_epoch": self.current_epoch + 1,
            "init_args": self.my_init_kwargs,
            "trainer_name": self.__class__.__name__,
            "inference_allowed_mirroring_axes": self.inference_allowed_mirroring_axes,
            "ich_config_sha256": self.ich_config_digest,
            "ich_config": {key: value for key, value in self.ich_config.items() if not key.startswith("_")},
            "ich_state": {
                "best_dice": float(self._best_dice),
                "best_recall": float(self._best_recall),
                "best_composite": float(self._best_composite),
                "best_full_macro_dice": float(self._best_full_macro_dice),
                "best_full_macro_epoch": self._best_full_macro_epoch,
                "pruning": self.early_stopping.state_dict(),
            },
        }
        path = Path(filename)
        temporary = path.with_name(path.name + ".tmp")
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)

    def load_checkpoint(self, filename_or_checkpoint) -> None:
        if isinstance(filename_or_checkpoint, (str, Path)):
            path = str(filename_or_checkpoint)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            saved_digest = checkpoint.get("ich_config_sha256")
            if saved_digest is not None and saved_digest != self.ich_config_digest:
                raise RuntimeError(
                    "checkpoint와 현재 ICH_NNUNET_CONFIG의 설정이 다릅니다. "
                    "학습 때 사용한 YAML을 지정하세요."
                )
            super().load_checkpoint(path)
        else:
            checkpoint = filename_or_checkpoint
            super().load_checkpoint(filename_or_checkpoint)
        state = checkpoint.get("ich_state", {})
        self._best_dice = float(state.get("best_dice", self._best_dice))
        self._best_recall = float(state.get("best_recall", self._best_recall))
        self._best_composite = float(state.get("best_composite", self._best_composite))
        self._best_full_macro_dice = float(state.get("best_full_macro_dice", self._best_full_macro_dice))
        self._best_full_macro_epoch = state.get("best_full_macro_epoch", self._best_full_macro_epoch)
        self.early_stopping.load_state_dict(state.get("pruning"))

    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ):
        return build_network(
            load_config()["network"],
            plans_manager,
            configuration_manager,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        )

    def _build_loss(self):
        return build_loss(self, self.ich_config["loss"])

    def get_dataloaders(self):
        if not self.ich_config.get("sampling", {}).get("enabled", False):
            return super().get_dataloaders()
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        patch_size = self.configuration_manager.patch_size
        scales = self._get_deep_supervision_scales()
        rotation, dummy_2d, initial_patch_size, mirror_axes = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        tr_transforms = self.get_training_transforms(
            patch_size, rotation, scales, mirror_axes, dummy_2d,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)
        val_transforms = self.get_validation_transforms(
            scales, is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        sampling_config = self.ich_config["sampling"]
        strategy = str(sampling_config.get("strategy", "case_weighted")).lower()
        loader_class, loader_extra, probabilities = nnUNetDataLoader, {}, None
        if strategy in {"component_center", "case_and_component"}:
            centers, component_report = build_component_centers(
                dataset_tr, self.configuration_manager.spacing, sampling_config)
            loader_class = ComponentCenteredDataLoader
            loader_extra["component_centers"] = centers
            if self.local_rank == 0:
                write_component_report(
                    Path(self.output_folder) / "component_sampling.json", component_report)
                self.print_to_log_file("Component-centered sampling enabled")
        if strategy in {"case_weighted", "case_and_component"}:
            probabilities, case_report = build_sampling_probabilities(
                dataset_tr, self.configuration_manager.spacing, sampling_config)
            if self.local_rank == 0:
                write_sampling_report(
                    Path(self.output_folder) / "small_lesion_sampling.json", case_report)
                self.print_to_log_file("Small-lesion case sampling enabled")
        dl_tr = loader_class(
            dataset_tr, self.batch_size, initial_patch_size, patch_size, self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=probabilities, pad_sides=None, transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling, **loader_extra)
        dl_val = nnUNetDataLoader(
            dataset_val, self.batch_size, patch_size, patch_size, self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling)
        allowed = get_allowed_n_proc_DA()
        if allowed == 0:
            train_generator = SingleThreadedAugmenter(dl_tr, None)
            val_generator = SingleThreadedAugmenter(dl_val, None)
        else:
            train_generator = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr, transform=None, num_processes=allowed, num_cached=max(6, allowed // 2),
                seeds=None, pin_memory=self.device.type == "cuda", wait_time=0.002)
            val_generator = NonDetMultiThreadedAugmenter(
                data_loader=dl_val, transform=None, num_processes=max(1, allowed // 2),
                num_cached=max(3, allowed // 4), seeds=None,
                pin_memory=self.device.type == "cuda", wait_time=0.002)
        _ = next(train_generator)
        _ = next(val_generator)
        return train_generator, val_generator

    def configure_optimizers(self):
        cfg = self.ich_config["trainer"]
        optimizer_cfg = cfg["optimizer"]
        name = str(optimizer_cfg["name"]).lower()
        if name == "sgd":
            optimizer = torch.optim.SGD(
                self.network.parameters(),
                lr=self.initial_lr,
                weight_decay=self.weight_decay,
                momentum=float(optimizer_cfg.get("momentum", 0.99)),
                nesterov=bool(optimizer_cfg.get("nesterov", True)),
            )
        elif name == "adamw":
            optimizer = torch.optim.AdamW(
                self.network.parameters(),
                lr=self.initial_lr,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"지원하지 않는 optimizer: {name}")

        scheduler_name = str(cfg["scheduler"]["name"]).lower()
        if scheduler_name != "poly":
            raise ValueError("nnU-Net baseline scheduler는 현재 poly만 지원합니다.")
        return optimizer, PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)

    def on_validation_epoch_end(self, val_outputs: list[dict]):
        super().on_validation_epoch_end(val_outputs)
        collated = collate_outputs(val_outputs)
        tp = np.sum(collated["tp_hard"], 0)
        fn = np.sum(collated["fn_hard"], 0)
        if self.is_ddp:
            gathered_tp = [None for _ in range(dist.get_world_size())]
            gathered_fn = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_tp, tp)
            dist.all_gather_object(gathered_fn, fn)
            tp = np.vstack(gathered_tp).sum(0)
            fn = np.vstack(gathered_fn).sum(0)
        denominator = tp + fn
        recalls = np.divide(tp, denominator, out=np.full_like(tp, np.nan, dtype=float), where=denominator > 0)
        finite_recalls = recalls[np.isfinite(recalls)]
        self._last_recall = float(finite_recalls.mean()) if finite_recalls.size else float("nan")

    def _load_full_validation_history(self) -> list[dict]:
        if not self._full_validation_history_path.is_file():
            return []
        history = json.loads(self._full_validation_history_path.read_text(encoding="utf-8"))
        if history:
            best = max(history, key=lambda item: float(item["macro_dice_positive"]))
            self._best_full_macro_dice = float(best["macro_dice_positive"])
            self._best_full_macro_epoch = int(best["epoch"])
            self._last_full_macro_dice = float(history[-1]["macro_dice_positive"])
        return history

    def _top_full_macro_epochs(self, history: list[dict] | None = None) -> list[int]:
        if history is None:
            history = self._load_full_validation_history()
        ranked = sorted(
            history,
            key=lambda item: float(item["macro_dice_positive"]),
            reverse=True,
        )
        return [int(item["epoch"]) for item in ranked[: self._full_validation_top_k]]

    def _retained_checkpoint_count(self) -> int:
        folder = Path(self.output_folder) / "epoch_checkpoints"
        return len(list(folder.glob("checkpoint_epoch_*.pth"))) if folder.is_dir() else 0

    def _prune_epoch_checkpoints(self, history: list[dict], completed_epochs: int) -> None:
        """Keep Top-K validation neighborhoods and the rolling unevaluated interval."""
        if not self._archive_every_epoch or self.local_rank != 0:
            return
        folder = Path(self.output_folder) / "epoch_checkpoints"
        if not folder.is_dir():
            return
        radius = self._full_validation_neighborhood_radius
        keep_epochs: set[int] = set()
        for anchor in self._top_full_macro_epochs(history):
            keep_epochs.update(range(max(1, anchor - radius), anchor + radius + 1))
        # Preserve the recent rolling window required by the next sampled epoch.
        keep_epochs.update(range(max(1, completed_epochs - radius + 1), completed_epochs + 1))
        for checkpoint in folder.glob("checkpoint_epoch_*.pth"):
            try:
                epoch = int(checkpoint.stem.rsplit("_", 1)[1])
            except (IndexError, ValueError):
                continue
            if epoch not in keep_epochs:
                checkpoint.unlink()

    def _archive_epoch_checkpoint(self, completed_epochs: int) -> Path | None:
        if not self._archive_every_epoch or self.local_rank != 0:
            return None
        archive_start_epoch = max(
            1,
            self._full_validation_start_epoch
            - self._full_validation_neighborhood_radius,
        )
        if completed_epochs < archive_start_epoch:
            return None
        folder = Path(self.output_folder) / "epoch_checkpoints"
        folder.mkdir(parents=True, exist_ok=True)
        destination = folder / f"checkpoint_epoch_{completed_epochs:04d}.pth"
        if destination.is_file():
            return destination
        self.current_epoch -= 1
        try:
            self.save_checkpoint(str(destination))
        finally:
            self.current_epoch += 1
        return destination

    @staticmethod
    def _positive_macro_from_summary(summary: dict) -> tuple[float, int]:
        values = []
        for item in summary.get("metric_per_case", []):
            metrics = item["metrics"]["1"]
            if float(metrics.get("n_ref", 0)) > 0:
                value = float(metrics["Dice"])
                if math.isfinite(value):
                    values.append(value)
        if not values:
            raise RuntimeError("full-volume validation has no positive cases")
        return float(np.mean(values)), len(values)

    def perform_actual_validation(self, save_probabilities: bool = False):
        """Run nnU-Net export in one thread to avoid tiny /dev/shm failures."""
        trainer_module = importlib.import_module(
            "nnunetv2.training.nnUNetTrainer.nnUNetTrainer"
        )
        original_multiprocessing = trainer_module.multiprocessing

        class _ThreadPoolContext:
            @staticmethod
            def Pool(processes=None):
                return ThreadPool(processes=1)

        class _ThreadedMultiprocessing:
            @staticmethod
            def get_context(method=None):
                return _ThreadPoolContext()

        trainer_module.multiprocessing = _ThreadedMultiprocessing
        try:
            return super().perform_actual_validation(save_probabilities)
        finally:
            trainer_module.multiprocessing = original_multiprocessing

    def _maybe_run_full_validation(self, completed_epochs: int, epoch_checkpoint: Path | None) -> None:
        if not self._full_validation_enabled:
            return
        if completed_epochs < self._full_validation_start_epoch:
            return
        if (
            completed_epochs - self._full_validation_start_epoch
        ) % self._full_validation_interval != 0:
            return
        if self.is_ddp:
            raise RuntimeError("periodic full-volume validation currently supports single-GPU training only")
        destination = Path(self.output_folder) / "full_validation" / f"epoch_{completed_epochs:04d}"
        summary_path = destination / "summary.json"
        if not summary_path.is_file():
            transient = Path(self.output_folder) / "validation"
            if transient.exists():
                shutil.rmtree(transient)
            self.print_to_log_file(f"Starting full-volume validation at epoch {completed_epochs}")
            self.perform_actual_validation(False)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                shutil.rmtree(destination)
            shutil.move(str(transient), str(destination))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        macro_dice, positive_cases = self._positive_macro_from_summary(summary)
        self._last_full_macro_dice = macro_dice
        history = self._load_full_validation_history()
        history = [item for item in history if int(item["epoch"]) != completed_epochs]
        record = {
            "epoch": completed_epochs,
            "macro_dice_positive": macro_dice,
            "positive_cases": positive_cases,
            "summary": str(summary_path),
            "checkpoint": str(epoch_checkpoint) if epoch_checkpoint else None,
        }
        history.append(record)
        history.sort(key=lambda item: int(item["epoch"]))
        self._write_json_atomic(self._full_validation_history_path, history)
        if macro_dice > self._best_full_macro_dice:
            self._best_full_macro_dice = macro_dice
            self._best_full_macro_epoch = completed_epochs
            best_path = Path(self.output_folder) / "checkpoint_best_full_macro.pth"
            if epoch_checkpoint is not None:
                shutil.copy2(epoch_checkpoint, best_path)
            else:
                self.current_epoch -= 1
                try:
                    self.save_checkpoint(str(best_path))
                finally:
                    self.current_epoch += 1
        self._prune_epoch_checkpoints(history, completed_epochs)
        self.network.train()
        self.print_to_log_file(
            f"Full-volume epoch {completed_epochs}: positive macro Dice={macro_dice:.6f}")

    def _save_metric_state(self, dice: float, recall: float, composite: float, epoch: int) -> None:
        if self.local_rank != 0:
            return

        payload = {
            "epoch": epoch,
            "train_loss": float(
                self.logger.get_value("train_losses", step=-1)
            ),
            "pseudo_dice": self._finite_or_none(dice),
            "pseudo_recall": self._finite_or_none(recall),
            "composite": self._finite_or_none(composite),
            "best_dice": self._finite_or_none(self._best_dice),
            "best_recall": self._finite_or_none(self._best_recall),
            "best_composite": self._finite_or_none(self._best_composite),
            "pruning_best": self._finite_or_none(self.early_stopping.best),
            "pruning_bad_checks": self.early_stopping.bad_checks,
            "metrics_note": "patch-based online validation metrics; final reporting requires full-volume evaluation",
        }
        self._write_json_atomic(self._metrics_path, payload)

    def on_epoch_end(self):
        epoch = self.current_epoch
        dice = float(self.logger.get_value("mean_fg_dice", step=-1))
        recall = float(self._last_recall)
        weights = self.ich_config["checkpoint"]
        composite = (
            float(weights["dice_weight"]) * dice + float(weights["recall_weight"]) * recall
            if math.isfinite(dice) and math.isfinite(recall)
            else float("nan")
        )
        self._last_dice = dice
        self._last_composite = composite
        min_delta = float(weights.get("min_delta", 0.0))
        improved_dice = math.isfinite(dice) and dice > self._best_dice + min_delta
        improved_recall = math.isfinite(recall) and recall > self._best_recall + min_delta
        improved_composite = math.isfinite(composite) and composite > self._best_composite + min_delta
        if improved_dice:
            self._best_dice = dice
        if improved_recall:
            self._best_recall = recall
        if improved_composite:
            self._best_composite = composite

        completed_epochs = epoch + 1
        if math.isfinite(composite):
            self._stop_requested = self.early_stopping.update(composite, completed_epochs)
        else:
            self._stop_requested = False
            self.print_to_log_file("ICH metrics: Recall/Composite가 NaN이라 best 저장과 pruning 갱신을 건너뜁니다.")
        self.print_to_log_file(
            f"ICH metrics: pseudo_dice={dice:.4f}, pseudo_recall={recall:.4f}, composite={composite:.4f}"
        )
        super().on_epoch_end()
        epoch_seconds = float(
            self.logger.get_value("epoch_end_timestamps", step=-1)
            - self.logger.get_value("epoch_start_timestamps", step=-1)
        )
        self._epoch_durations.append(epoch_seconds)
        if improved_dice or improved_recall or improved_composite:
            self.current_epoch -= 1
            try:
                if improved_dice:
                    self.save_checkpoint(str(Path(self.output_folder) / "checkpoint_best_dice.pth"))
                if improved_recall:
                    self.save_checkpoint(str(Path(self.output_folder) / "checkpoint_best_recall.pth"))
                if improved_composite:
                    self.save_checkpoint(str(Path(self.output_folder) / "checkpoint_best_composite.pth"))
            finally:
                self.current_epoch += 1
        epoch_checkpoint = self._archive_epoch_checkpoint(completed_epochs)
        self._maybe_run_full_validation(completed_epochs, epoch_checkpoint)
        self._save_metric_state(dice, recall, composite, epoch)
        self._write_progress(
            "early_stopping" if self._stop_requested else "running",
            epoch + 1,
            dice,
            recall,
            composite,
        )

    def _run_training_without_progress_guard(self):
        self.on_train_start()
        for _ in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()
            self.on_train_epoch_start()
            train_outputs = [self.train_step(next(self.dataloader_train)) for _ in range(self.num_iterations_per_epoch)]
            self.on_train_epoch_end(train_outputs)
            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = [self.validation_step(next(self.dataloader_val)) for _ in range(self.num_val_iterations_per_epoch)]
                self.on_validation_epoch_end(val_outputs)
            self.on_epoch_end()
            if self._stop_requested:
                self.print_to_log_file("Pruning: validation composite가 patience 동안 개선되지 않아 조기 종료합니다.")
                break
        self.on_train_end()

    def run_training(self):
        try:
            self.on_train_start()
            for _ in range(self.current_epoch, self.num_epochs):
                self.on_epoch_start()
                self.on_train_epoch_start()
                train_outputs = [
                    self.train_step(next(self.dataloader_train))
                    for _ in range(self.num_iterations_per_epoch)
                ]
                self.on_train_epoch_end(train_outputs)
                with torch.no_grad():
                    self.on_validation_epoch_start()
                    val_outputs = [
                        self.validation_step(next(self.dataloader_val))
                        for _ in range(self.num_val_iterations_per_epoch)
                    ]
                    self.on_validation_epoch_end(val_outputs)
                self.on_epoch_end()
                if self._stop_requested:
                    self.print_to_log_file(
                        "Pruning: composite가 patience 동안 개선되지 않아 조기 종료합니다."
                    )
                    break
            self.on_train_end()
            self._write_progress(
                "early_stopped" if self._stop_requested else "completed",
                self.current_epoch,
            )
        except BaseException as exc:
            self._write_progress("failed", self.current_epoch, message=repr(exc))
            raise
