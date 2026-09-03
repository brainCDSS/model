from __future__ import annotations

import csv
import gc
import json
import os
import random
import shutil
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from nnunetv2.inference.export_prediction import (
    export_prediction_from_logits,
)
from nnunetv2.inference.predict_from_raw_data import (
    nnUNetPredictor,
)
from nnunetv2.inference.sliding_window_prediction import (
    compute_gaussian,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import (
    nnUNetTrainer,
)
from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.utilities.label_handling.label_handling import (
    convert_labelmap_to_one_hot,
)


class nnUNetTrainerAneurysmEarlyStopping(nnUNetTrainer):
    """
    A-N00 Standard nnU-Net baseline trainer.

    Network architecture, loss, optimizer type, augmentation and
    deep supervision remain the Standard nnU-Net implementation.

    A-N00 specific policy
    ---------------------
    1. Maximum 300 epochs
    2. 250 train iterations / epoch
    3. 50 online patch-validation iterations / epoch
       - diagnostic only
       - NOT used for best checkpoint selection
    4. Full-volume validation on all validation patients every epoch
    5. Best checkpoint criterion:
       val_patient_mean_dice
    6. EarlyStopping criterion:
       val_patient_mean_dice
    7. patience=30, min_delta=0.0001
    8. Per-patient / per-class / epoch summary metrics are stored
       as CSV/JSON.
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            device=device,
        )

        # =====================================================
        # TRAINING POLICY
        # =====================================================

        self.num_epochs = int(
            os.environ.get(
                "A_N00_MAX_EPOCHS",
                "300",
            )
        )

        self.initial_lr = float(
            os.environ.get(
                "A_N00_INITIAL_LR",
                "0.01",
            )
        )

        self.weight_decay = float(
            os.environ.get(
                "A_N00_WEIGHT_DECAY",
                "0.00003",
            )
        )

        self.oversample_foreground_percent = float(
            os.environ.get(
                "A_N00_OVERSAMPLE",
                "0.33",
            )
        )

        self.num_iterations_per_epoch = int(
            os.environ.get(
                "A_N00_TRAIN_ITERS",
                "250",
            )
        )

        self.num_val_iterations_per_epoch = int(
            os.environ.get(
                "A_N00_VAL_ITERS",
                "50",
            )
        )

        # =====================================================
        # CHECKPOINT / PROGRESS
        # =====================================================

        self.save_every = int(
            os.environ.get(
                "A_N00_SAVE_EVERY",
                "1",
            )
        )

        self.progress_every = max(
            1,
            int(
                os.environ.get(
                    "A_N00_PROGRESS_EVERY",
                    "25",
                )
            ),
        )

        self.patient_progress_every = max(
            1,
            int(
                os.environ.get(
                    "A_N00_PATIENT_PROGRESS_EVERY",
                    "1",
                )
            ),
        )

        self.expected_validation_cases = int(
            os.environ.get(
                "A_N00_EXPECTED_VAL_CASES",
                "46",
            )
        )

        # =====================================================
        # EARLY STOPPING
        # =====================================================

        # A_N00_FINAL_POLICY_V2
        #
        # Full-volume Patient Validation is NOT performed
        # during training epochs.
        #
        # Best / EarlyStopping use nnU-Net online patch
        # validation EMA Dice.
        self.early_stopping_monitor = (
            "ema_fg_dice"
        )

        self.early_stopping_patience = int(
            os.environ.get(
                "A_N00_PATIENCE",
                "30",
            )
        )

        self.early_stopping_min_delta = float(
            os.environ.get(
                "A_N00_MIN_DELTA",
                "0.0001",
            )
        )

        self.early_stopping_start_epoch = int(
            os.environ.get(
                "A_N00_EARLY_STOP_START",
                "0",
            )
        )

        # =====================================================
        # PATIENT VALIDATION STATE
        # =====================================================

        self._latest_patient_metrics = None
        self._latest_patient_rows = None
        self._latest_class_rows = None

        # Full-volume Patient Mean Dice is a FINAL reporting
        # metric only. It no longer selects training checkpoints.
        self._best_patient_mean_dice = None

        # =====================================================
        # A_N00_FINAL_POLICY_V2
        # Strict interruption resume
        # =====================================================

        self._resume_phase = "epoch_start"

        self._resume_train_next_iteration = 0

        self._resume_train_losses = []

        # load_checkpoint happens before on_train_start when
        # --c is used. on_train_start creates dataloaders and
        # consumes RNG once. Therefore RNG is restored AFTER
        # on_train_start.
        self._pending_rng_state = None

        # Final Full-volume Validation metadata
        self._final_selected_best_epoch_1based = None
        self._final_selected_best_ema = None

    # =========================================================
    # PATHS
    # =========================================================

    @property
    def patient_validation_root(self) -> Path:
        return (
            Path(self.output_folder)
            / "patient_validation"
        )

    @property
    def training_history_file(self) -> Path:
        return (
            self.patient_validation_root
            / "training_history.csv"
        )

    # =========================================================
    # ATOMIC CHECKPOINT
    # =========================================================

    def save_checkpoint(
        self,
        filename: str,
    ) -> None:
        """
        Keep the official nnU-Net checkpoint structure.

        Save to a temporary path first and atomically replace
        the destination to reduce corruption risk when the
        runtime is interrupted.
        """

        tmp_filename = (
            f"{filename}.tmp"
        )

        super().save_checkpoint(
            tmp_filename
        )

        if self.local_rank == 0:
            tmp_path = Path(
                tmp_filename
            )

            if tmp_path.exists():
                os.replace(
                    tmp_filename,
                    filename,
                )


    # =========================================================
    # A_N00_FINAL_POLICY_V2
    # STRICT INTERRUPTION RESUME
    # =========================================================

    @property
    def interruption_resume_root(
        self,
    ) -> Path:

        path = (
            Path(
                self.output_folder
            )
            / "interruption_resume"
        )

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        return path


    @property
    def final_validation_root(
        self,
    ) -> Path:

        path = (
            Path(
                self.output_folder
            )
            / "final_validation"
        )

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        return path


    @property
    def final_validation_resume_file(
        self,
    ) -> Path:

        return (
            self.final_validation_root
            / "resume_state.pth"
        )


    @property
    def final_validation_complete_file(
        self,
    ) -> Path:

        return (
            self.final_validation_root
            / "COMPLETE.json"
        )


    def _online_val_resume_file(
        self,
        epoch=None,
    ) -> Path:

        if epoch is None:

            epoch = (
                self.current_epoch
            )

        return (
            self.interruption_resume_root
            / (
                f"epoch_{epoch + 1:03d}"
                "_online_val.pth"
            )
        )


    @staticmethod
    def _atomic_torch_save(
        payload,
        destination,
    ) -> None:

        destination = Path(
            destination
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = Path(
            str(destination)
            + ".tmp"
        )

        with temporary.open(
            "wb"
        ) as f:

            torch.save(
                payload,
                f,
            )

            f.flush()

            try:

                os.fsync(
                    f.fileno()
                )

            except OSError:

                pass

        os.replace(
            temporary,
            destination,
        )


    @staticmethod
    def _capture_rng_state():

        state = {
            "python":
                random.getstate(),

            "numpy":
                np.random.get_state(),

            "torch_cpu":
                torch.get_rng_state(),

            "torch_cuda":
                None,
        }

        if torch.cuda.is_available():

            state[
                "torch_cuda"
            ] = (
                torch.cuda.get_rng_state_all()
            )

        return state


    @staticmethod
    def _restore_rng_state(
        state,
    ) -> None:

        if not state:

            return

        random.setstate(
            state[
                "python"
            ]
        )

        np.random.set_state(
            state[
                "numpy"
            ]
        )

        _a_n00_cpu_rng_state = state[
                "torch_cpu"
            ]

        if not isinstance(
            _a_n00_cpu_rng_state,
            torch.Tensor,
        ):
            _a_n00_cpu_rng_state = torch.as_tensor(
                _a_n00_cpu_rng_state,
                dtype=torch.uint8,
                device="cpu",
            )
        else:
            _a_n00_cpu_rng_state = (
                _a_n00_cpu_rng_state
                .detach()
                .to(
                    device="cpu",
                    dtype=torch.uint8,
                )
            )

        torch.set_rng_state(
            _a_n00_cpu_rng_state
        )

        cuda_state = (
            state.get(
                "torch_cuda"
            )
        )

        if (
            cuda_state is not None
            and torch.cuda.is_available()
        ):

            # =================================================
            # A-N00 CUDA RNG strict-resume compatibility
            #
            # torch.cuda.set_rng_state_all() requires:
            #     list[torch.ByteTensor]
            #
            # Runtime/serialization 과정에서
            # Tensor / ndarray / list[int] 등으로 복원되어도
            # 항상 CPU uint8 ByteTensor list로 정규화한다.
            # =================================================

            if isinstance(
                cuda_state,
                torch.Tensor,
            ):

                cuda_tensor = (
                    cuda_state
                    .detach()
                    .to(
                        device="cpu",
                        dtype=torch.uint8,
                    )
                )

                if cuda_tensor.ndim == 1:

                    cuda_states = [
                        cuda_tensor
                        .reshape(-1)
                        .contiguous()
                    ]

                elif cuda_tensor.ndim == 2:

                    cuda_states = [
                        row
                        .reshape(-1)
                        .contiguous()
                        for row in cuda_tensor
                    ]

                else:

                    raise RuntimeError(
                        "Unsupported saved CUDA RNG tensor shape: "
                        f"{tuple(cuda_tensor.shape)}"
                    )

            elif isinstance(
                cuda_state,
                np.ndarray,
            ):

                if cuda_state.ndim == 1:

                    cuda_states = [
                        torch.as_tensor(
                            cuda_state,
                            dtype=torch.uint8,
                            device="cpu",
                        )
                        .reshape(-1)
                        .contiguous()
                    ]

                elif cuda_state.ndim == 2:

                    cuda_states = [
                        torch.as_tensor(
                            row,
                            dtype=torch.uint8,
                            device="cpu",
                        )
                        .reshape(-1)
                        .contiguous()
                        for row in cuda_state
                    ]

                else:

                    raise RuntimeError(
                        "Unsupported saved CUDA RNG ndarray shape: "
                        f"{cuda_state.shape}"
                    )

            elif isinstance(
                cuda_state,
                (list, tuple),
            ):

                if len(cuda_state) == 0:

                    raise RuntimeError(
                        "Saved CUDA RNG state is empty."
                    )

                # [1, 2, 3, ...]
                # 또는 [tensor(1), tensor(2), ...]
                # = 단일 GPU RNG state가 list 형태로 직렬화된 경우
                if all(
                    isinstance(
                        value,
                        (int, np.integer),
                    )
                    or (
                        isinstance(
                            value,
                            torch.Tensor,
                        )
                        and value.ndim == 0
                    )
                    for value in cuda_state
                ):

                    cuda_states = [
                        torch.tensor(
                            [
                                int(
                                    value.item()
                                )
                                if isinstance(
                                    value,
                                    torch.Tensor,
                                )
                                else int(value)
                                for value
                                in cuda_state
                            ],
                            dtype=torch.uint8,
                            device="cpu",
                        )
                        .reshape(-1)
                        .contiguous()
                    ]

                # 정상 multi-GPU 형식:
                # [state_gpu0, state_gpu1, ...]
                else:

                    cuda_states = []

                    for value in cuda_state:

                        if isinstance(
                            value,
                            torch.Tensor,
                        ):

                            fixed = (
                                value
                                .detach()
                                .to(
                                    device="cpu",
                                    dtype=torch.uint8,
                                )
                                .reshape(-1)
                                .contiguous()
                            )

                        elif isinstance(
                            value,
                            np.ndarray,
                        ):

                            fixed = (
                                torch.as_tensor(
                                    value,
                                    dtype=torch.uint8,
                                    device="cpu",
                                )
                                .reshape(-1)
                                .contiguous()
                            )

                        else:

                            fixed = (
                                torch.as_tensor(
                                    value,
                                    dtype=torch.uint8,
                                    device="cpu",
                                )
                                .reshape(-1)
                                .contiguous()
                            )

                        cuda_states.append(
                            fixed
                        )

            else:

                raise TypeError(
                    "Unsupported saved CUDA RNG type: "
                    f"{type(cuda_state)}"
                )

            # 마지막 형식 강제 보장
            cuda_states = [
                value
                .detach()
                .to(
                    device="cpu",
                    dtype=torch.uint8,
                )
                .reshape(-1)
                .contiguous()
                for value in cuda_states
            ]

            gpu_count = (
                torch.cuda.device_count()
            )

            if len(cuda_states) != gpu_count:

                raise RuntimeError(
                    "CUDA RNG GPU count mismatch: "
                    f"saved={len(cuda_states)}, "
                    f"current={gpu_count}"
                )

            torch.cuda.set_rng_state_all(
                cuda_states
            )

            print(
                "[RESUME] CUDA RNG restored "
                f"for {gpu_count} GPU(s)",
                flush=True,
            )


    def _restore_pending_rng_state(
        self,
    ) -> None:

        if (
            self._pending_rng_state
            is None
        ):

            return

        self._restore_rng_state(
            self._pending_rng_state
        )

        self._pending_rng_state = None

        self.print_to_log_file(
            "[RESUME] RNG state restored "
            "after dataloader initialization"
        )


    def _network_module_for_checkpoint(
        self,
    ):

        if self.is_ddp:

            mod = (
                self.network.module
            )

        else:

            mod = (
                self.network
            )

        # nnUNet_compile=false for A-N00,
        # but retain compatibility.
        if hasattr(
            mod,
            "_orig_mod",
        ):

            mod = (
                mod._orig_mod
            )

        return mod


    def _save_strict_resume_checkpoint(
        self,
        phase: str,
        train_next_iteration: int = 0,
        train_losses=None,
    ) -> None:
        """
        Mid-epoch checkpoint.

        IMPORTANT:
        Unlike official nnU-Net save_checkpoint(),
        current_epoch is NOT incremented here.

        Example:
          current_epoch = 7
          completed iteration = 183

        checkpoint:
          current_epoch = 7
          next_iteration = 183

        resume:
          Epoch 8
          next displayed iteration = 184/250
        """

        if (
            self.local_rank != 0
            or self.disable_checkpointing
        ):

            return

        allowed = {
            "train",
            "online_val",
            "epoch_end",
        }

        if phase not in allowed:

            raise ValueError(
                f"Invalid strict-resume phase: {phase}"
            )

        losses = (
            []
            if train_losses is None
            else [
                float(x)
                for x
                in train_losses
            ]
        )

        mod = (
            self._network_module_for_checkpoint()
        )

        checkpoint = {
            # -----------------------------------------
            # Official nnU-Net checkpoint fields
            # -----------------------------------------

            "network_weights":
                mod.state_dict(),

            "optimizer_state":
                self.optimizer.state_dict(),

            "grad_scaler_state":
                (
                    self.grad_scaler.state_dict()
                    if self.grad_scaler
                    is not None
                    else None
                ),

            "logging":
                self.logger.get_checkpoint(),

            "_best_ema":
                self._best_ema,

            # CRITICAL:
            # same epoch, not +1
            "current_epoch":
                int(
                    self.current_epoch
                ),

            "init_args":
                self.my_init_kwargs,

            "trainer_name":
                self.__class__.__name__,

            "inference_allowed_mirroring_axes":
                self.inference_allowed_mirroring_axes,

            # -----------------------------------------
            # A-N00 custom fields
            # -----------------------------------------

            "a_n00_final_policy_version":
                2,

            "a_n00_resume_phase":
                phase,

            "a_n00_train_next_iteration":
                int(
                    train_next_iteration
                ),

            "a_n00_train_losses":
                losses,

            "a_n00_rng_state":
                self._capture_rng_state(),
        }

        destination = (
            Path(
                self.output_folder
            )
            / "checkpoint_latest.pth"
        )

        self._atomic_torch_save(
            checkpoint,
            destination,
        )


    def load_checkpoint(
        self,
        filename_or_checkpoint,
    ) -> None:
        """
        Load both:
          - official epoch-complete nnU-Net checkpoint
          - A-N00 strict mid-epoch checkpoint
        """

        temporary_file = None

        if isinstance(
            filename_or_checkpoint,
            (str, Path),
        ):

            checkpoint_path = Path(
                filename_or_checkpoint
            )

            checkpoint = torch.load(
                str(
                    checkpoint_path
                ),
                map_location=self.device,
                weights_only=False,
            )

            # Base nnU-Net expects a normal path.
            super().load_checkpoint(
                str(
                    checkpoint_path
                )
            )

        else:

            checkpoint = (
                filename_or_checkpoint
            )

            # Make dictionary loading compatible with
            # nnU-Net 2.8.1 path-oriented implementation.
            with tempfile.NamedTemporaryFile(
                suffix=".pth",
                delete=False,
            ) as f:

                temporary_file = Path(
                    f.name
                )

            try:

                torch.save(
                    checkpoint,
                    temporary_file,
                )

                super().load_checkpoint(
                    str(
                        temporary_file
                    )
                )

            finally:

                try:

                    temporary_file.unlink()

                except (
                    FileNotFoundError,
                    AttributeError,
                ):

                    pass


        if (
            checkpoint.get(
                "a_n00_final_policy_version"
            )
            == 2
        ):

            self._resume_phase = (
                checkpoint.get(
                    "a_n00_resume_phase",
                    "epoch_start",
                )
            )

            self._resume_train_next_iteration = int(
                checkpoint.get(
                    "a_n00_train_next_iteration",
                    0,
                )
            )

            self._resume_train_losses = [
                float(x)
                for x
                in checkpoint.get(
                    "a_n00_train_losses",
                    [],
                )
            ]

            self._pending_rng_state = (
                checkpoint.get(
                    "a_n00_rng_state"
                )
            )

            self.print_to_log_file(
                "[RESUME LOAD] "
                f"epoch="
                f"{self.current_epoch + 1} | "
                f"phase="
                f"{self._resume_phase} | "
                f"train_completed="
                f"{self._resume_train_next_iteration}/"
                f"{self.num_iterations_per_epoch}"
            )

        else:

            # Official epoch-complete checkpoint.
            self._resume_phase = (
                "epoch_start"
            )

            self._resume_train_next_iteration = 0

            self._resume_train_losses = []

            self._pending_rng_state = None


    # ---------------------------------------------------------
    # Online patch-validation iteration resume
    # ---------------------------------------------------------

    def _save_online_val_resume_state(
        self,
        next_iteration: int,
        val_outputs,
    ) -> None:

        if self.local_rank != 0:

            return

        payload = {
            "epoch":
                int(
                    self.current_epoch
                ),

            "next_iteration":
                int(
                    next_iteration
                ),

            "val_outputs":
                val_outputs,

            "rng_state":
                self._capture_rng_state(),
        }

        self._atomic_torch_save(
            payload,
            self._online_val_resume_file(),
        )


    def _load_online_val_resume_state(
        self,
    ):

        path = (
            self._online_val_resume_file()
        )

        if not path.is_file():

            return None

        state = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        if int(
            state.get(
                "epoch",
                -1,
            )
        ) != int(
            self.current_epoch
        ):

            return None

        return state


    def _clear_online_val_resume_state(
        self,
        epoch_0based,
    ) -> None:

        path = (
            self._online_val_resume_file(
                epoch_0based
            )
        )

        try:

            path.unlink()

        except FileNotFoundError:

            pass


    # ---------------------------------------------------------
    # Final 46-patient validation resume
    # ---------------------------------------------------------

    def _save_final_patient_resume_state(
        self,
        patient_rows,
        class_counts,
        negative_cases,
        negative_case_fp,
        completed_case_ids,
    ) -> None:

        if self.local_rank != 0:

            return

        payload = {
            "policy_version":
                2,

            "selected_best_epoch_1based":
                int(
                    self._final_selected_best_epoch_1based
                ),

            "selected_best_ema":
                (
                    None
                    if (
                        self._final_selected_best_ema
                        is None
                    )
                    else float(
                        self._final_selected_best_ema
                    )
                ),

            "patient_rows":
                patient_rows,

            "class_counts":
                class_counts,

            "negative_cases":
                int(
                    negative_cases
                ),

            "negative_case_fp":
                int(
                    negative_case_fp
                ),

            "completed_case_ids":
                list(
                    completed_case_ids
                ),
        }

        self._atomic_torch_save(
            payload,
            self.final_validation_resume_file,
        )


    def _load_final_patient_resume_state(
        self,
    ):

        path = (
            self.final_validation_resume_file
        )

        if not path.is_file():

            return None

        state = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        if (
            int(
                state.get(
                    "policy_version",
                    -1,
                )
            )
            != 2
        ):

            return None

        state_epoch = int(
            state.get(
                "selected_best_epoch_1based",
                -1,
            )
        )

        if (
            state_epoch
            != int(
                self._final_selected_best_epoch_1based
            )
        ):

            raise RuntimeError(
                "Final Patient Validation resume state의 "
                "Best epoch이 현재 checkpoint_best와 다릅니다."
            )

        return state


    def _clear_final_patient_resume_state(
        self,
    ) -> None:

        try:

            self.final_validation_resume_file.unlink()

        except FileNotFoundError:

            pass


    # ---------------------------------------------------------
    # Final metric output
    # ---------------------------------------------------------

    def _write_final_validation_outputs(
        self,
        metrics,
        patient_rows,
        class_rows,
    ) -> None:

        if self.local_rank != 0:

            return

        root = (
            self.final_validation_root
        )

        self._write_csv(
            root
            / "per_patient_metrics.csv",
            patient_rows,
        )

        self._write_csv(
            root
            / "per_class_metrics.csv",
            class_rows,
        )

        summary = {
            key:
                self._json_value(
                    value
                )
            for key, value
            in metrics.items()
        }

        summary.update(
            {
                "validation_policy":
                    "once_after_training",

                "validated_checkpoint":
                    "checkpoint_best.pth",

                "checkpoint_selection_monitor":
                    "ema_fg_dice",

                "selected_best_epoch_1based":
                    int(
                        self._final_selected_best_epoch_1based
                    ),

                "selected_best_ema_fg_dice":
                    self._json_value(
                        self._final_selected_best_ema
                    ),

                "expected_validation_cases":
                    int(
                        self.expected_validation_cases
                    ),

                "patient_macro_policy":
                    (
                        "GT-present Location classes "
                        "per patient only"
                    ),
            }
        )

        summary_tmp = (
            root
            / "summary.json.tmp"
        )

        summary_file = (
            root
            / "summary.json"
        )

        with summary_tmp.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                summary,
                f,
                indent=2,
                ensure_ascii=False,
            )

            f.flush()

        os.replace(
            summary_tmp,
            summary_file,
        )

        # COMPLETE is written LAST.
        # Its presence means all final CSV/JSON outputs exist.
        complete_tmp = (
            root
            / "COMPLETE.json.tmp"
        )

        complete_file = (
            root
            / "COMPLETE.json"
        )

        complete_payload = {
            "status":
                "COMPLETE",

            "validated_checkpoint":
                "checkpoint_best.pth",

            "selected_best_epoch_1based":
                int(
                    self._final_selected_best_epoch_1based
                ),

            "selected_best_ema_fg_dice":
                self._json_value(
                    self._final_selected_best_ema
                ),

            "validation_cases":
                len(
                    patient_rows
                ),
        }

        with complete_tmp.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                complete_payload,
                f,
                indent=2,
                ensure_ascii=False,
            )

            f.flush()

        os.replace(
            complete_tmp,
            complete_file,
        )


    # =========================================================
    # METRIC HELPERS
    # =========================================================

    @staticmethod
    def _finite_mean(
        values,
    ) -> float:
        arr = np.asarray(
            values,
            dtype=np.float64,
        )

        finite = np.isfinite(
            arr
        )

        if not finite.any():
            return float("nan")

        return float(
            np.mean(
                arr[finite]
            )
        )

    @staticmethod
    def _finite_median(
        values,
    ) -> float:
        arr = np.asarray(
            values,
            dtype=np.float64,
        )

        finite = np.isfinite(
            arr
        )

        if not finite.any():
            return float("nan")

        return float(
            np.median(
                arr[finite]
            )
        )

    @staticmethod
    def _patient_binary_metrics(
        gt: np.ndarray,
        pred: np.ndarray,
    ) -> dict:
        """
        Merge all foreground labels into one foreground mask.

        Policy
        ------
        GT empty + Pred empty:
            Dice/Precision/Recall = NaN
            -> excluded from patient means.

        GT empty + Pred positive:
            Dice = 0
            Precision = 0
            Recall = NaN
            -> false-positive negative case.

        GT positive:
            Standard binary foreground metrics.
        """

        gt_fg = (
            gt > 0
        )

        pred_fg = (
            pred > 0
        )

        gt_voxels = int(
            np.count_nonzero(
                gt_fg
            )
        )

        pred_voxels = int(
            np.count_nonzero(
                pred_fg
            )
        )

        tp = int(
            np.count_nonzero(
                gt_fg & pred_fg
            )
        )

        fp = int(
            np.count_nonzero(
                (~gt_fg) & pred_fg
            )
        )

        fn = int(
            np.count_nonzero(
                gt_fg & (~pred_fg)
            )
        )

        if gt_voxels == 0:

            if pred_voxels == 0:
                dice = float("nan")
                precision = float("nan")
                recall = float("nan")

            else:
                dice = 0.0
                precision = 0.0
                recall = float("nan")

        else:
            dice_denominator = (
                2 * tp
                + fp
                + fn
            )

            dice = (
                float(
                    2 * tp
                    / dice_denominator
                )
                if dice_denominator > 0
                else 0.0
            )

            precision_denominator = (
                tp + fp
            )

            precision = (
                float(
                    tp
                    / precision_denominator
                )
                if precision_denominator > 0
                else 0.0
            )

            recall_denominator = (
                tp + fn
            )

            recall = (
                float(
                    tp
                    / recall_denominator
                )
                if recall_denominator > 0
                else 0.0
            )

        return {
            "gt_foreground_voxels":
                gt_voxels,

            "pred_foreground_voxels":
                pred_voxels,

            "tp":
                tp,

            "fp":
                fp,

            "fn":
                fn,

            "dice":
                dice,

            "precision":
                precision,

            "recall":
                recall,

            "is_negative_case":
                int(
                    gt_voxels == 0
                ),

            "false_positive_on_negative":
                int(
                    gt_voxels == 0
                    and pred_voxels > 0
                ),
        }

    @staticmethod
    def _format_metric(
        value,
    ) -> str:
        if (
            value is None
            or not np.isfinite(
                value
            )
        ):
            return "NaN"

        return (
            f"{float(value):.6f}"
        )

    # =========================================================
    # CSV / JSON
    # =========================================================

    @staticmethod
    def _write_csv(
        path: Path,
        rows: list[dict],
    ) -> None:

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not rows:
            return

        fieldnames = list(
            rows[0].keys()
        )

        with path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(
                rows
            )

    @staticmethod
    def _json_value(
        value,
    ):
        """
        JSON does not have a standard NaN representation.
        Convert non-finite floating values to null.
        """

        if isinstance(
            value,
            (float, np.floating),
        ):
            if not np.isfinite(
                value
            ):
                return None

            return float(
                value
            )

        if isinstance(
            value,
            (np.integer,),
        ):
            return int(
                value
            )

        return value

    def _write_epoch_outputs(
        self,
        metrics: dict,
        patient_rows: list[dict],
        class_rows: list[dict],
        is_best: bool,
    ) -> None:

        if self.local_rank != 0:
            return

        epoch_number = (
            self.current_epoch + 1
        )

        epoch_dir = (
            self.patient_validation_root
            / f"epoch_{epoch_number:03d}"
        )

        epoch_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._write_csv(
            epoch_dir
            / "per_patient_metrics.csv",
            patient_rows,
        )

        self._write_csv(
            epoch_dir
            / "per_class_metrics.csv",
            class_rows,
        )

        summary = {
            key: self._json_value(
                value
            )
            for key, value
            in metrics.items()
        }

        summary.update(
            {
                "epoch_0based":
                    self.current_epoch,

                "epoch_1based":
                    epoch_number,

                "monitor":
                    self.early_stopping_monitor,

                "is_best":
                    bool(
                        is_best
                    ),

                "best_patient_mean_dice":
                    self._json_value(
                        self._best_patient_mean_dice
                    ),
            }
        )

        with (
            epoch_dir
            / "summary.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                summary,
                f,
                indent=2,
                ensure_ascii=False,
            )

    def _upsert_training_history(
        self,
        row: dict,
    ) -> None:
        """
        Keep one row per epoch.

        If an interrupted runtime reruns the same epoch after
        resume, replace that epoch instead of duplicating it.
        """

        if self.local_rank != 0:
            return

        path = (
            self.training_history_file
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        rows = []

        if path.exists():

            with path.open(
                "r",
                newline="",
                encoding="utf-8",
            ) as f:

                reader = csv.DictReader(
                    f
                )

                rows = list(
                    reader
                )

        epoch_key = str(
            row["epoch_0based"]
        )

        rows = [
            existing
            for existing in rows
            if str(
                existing.get(
                    "epoch_0based"
                )
            )
            != epoch_key
        ]

        rows.append(
            {
                key: (
                    ""
                    if (
                        isinstance(
                            value,
                            (float, np.floating),
                        )
                        and not np.isfinite(
                            value
                        )
                    )
                    else value
                )
                for key, value
                in row.items()
            }
        )

        rows.sort(
            key=lambda x: int(
                x[
                    "epoch_0based"
                ]
            )
        )

        fieldnames = list(
            row.keys()
        )

        with path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for existing in rows:

                writer.writerow(
                    {
                        key:
                            existing.get(
                                key,
                                "",
                            )
                        for key
                        in fieldnames
                    }
                )

    # =========================================================
    # RESUME STATE
    # =========================================================

    def _restore_ema_best_from_history(
        self,
    ) -> None:
        """
        Safety reconciliation between CSV history and checkpoint.
        """

        path = (
            self.training_history_file
        )

        if not path.exists():

            return

        scores = []

        with path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as f:

            reader = csv.DictReader(
                f
            )

            for row in reader:

                raw = row.get(
                    "online_patch_ema_fg_dice",
                    "",
                )

                if raw in {
                    "",
                    None,
                }:

                    continue

                try:

                    value = float(
                        raw
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    continue

                if np.isfinite(
                    value
                ):

                    scores.append(
                        value
                    )

        if scores:

            history_best = max(
                scores
            )

            if (
                self._best_ema is None
                or history_best
                > self._best_ema
            ):

                self._best_ema = (
                    history_best
                )


    # =========================================================
    # EARLY STOPPING
    # =========================================================

    def _monitor_history(
        self,
    ) -> list[float]:

        path = (
            self.training_history_file
        )

        if not path.exists():

            return []

        history = []

        with path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as f:

            reader = csv.DictReader(
                f
            )

            for row in reader:

                raw_epoch = row.get(
                    "epoch_0based",
                    "",
                )

                raw_score = row.get(
                    "online_patch_ema_fg_dice",
                    "",
                )

                try:

                    epoch = int(
                        raw_epoch
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    continue

                if (
                    epoch
                    < self.early_stopping_start_epoch
                ):

                    continue

                try:

                    score = float(
                        raw_score
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    score = float(
                        "nan"
                    )

                history.append(
                    score
                )

        return history


    def _calculate_early_stopping_state(
        self,
    ):
        """
        min_delta affects the EarlyStopping reset criterion.

        Best checkpoint represents the true maximum
        online patch ema_fg_dice observed so far.
        """

        history = (
            self._monitor_history()
        )

        best = None
        bad_epochs = 0

        for raw_score in history:

            if not np.isfinite(
                raw_score
            ):
                bad_epochs += 1
                continue

            score = float(
                raw_score
            )

            if best is None:
                best = score
                bad_epochs = 0
                continue

            improved = (
                score
                > best
                + self.early_stopping_min_delta
            )

            if improved:
                best = score
                bad_epochs = 0

            else:
                bad_epochs += 1

        return (
            best,
            bad_epochs,
        )

    def _should_early_stop(
        self,
    ) -> bool:

        completed_epoch = (
            self.current_epoch - 1
        )

        if (
            completed_epoch
            < self.early_stopping_start_epoch
        ):
            return False

        best, bad_epochs = (
            self._calculate_early_stopping_state()
        )

        latest_history = (
            self._monitor_history()
        )

        latest = (
            latest_history[-1]
            if latest_history
            else None
        )

        self.print_to_log_file(
            "[EarlyStopping] "
            f"monitor={self.early_stopping_monitor} | "
            f"latest={self._format_metric(latest)} | "
            f"best(min_delta)={self._format_metric(best)} | "
            f"no_improvement={bad_epochs}/"
            f"{self.early_stopping_patience}"
        )

        return (
            bad_epochs
            >= self.early_stopping_patience
        )

    # =========================================================
    # FULL-VOLUME PATIENT VALIDATION
    # =========================================================


    # =========================================================
    # A_N00_MEMORY_SAFE_FINAL_EXPORT_V2
    #
    # 53-class logits 전체를 한꺼번에 original shape으로
    # resample하지 않는다.
    #
    # 각 class를 동일 nnU-Net resampling 함수로 순차 처리하고
    # running argmax를 수행한다.
    #
    # Dataset501은 class 0~52 contiguous multiclass임.
    # =========================================================

    def _export_prediction_memory_safe(
        self,
        prediction_logits,
        properties,
        output_filename_truncated,
    ) -> None:

        if self.label_manager.has_regions:
            raise RuntimeError(
                "A-N00 memory-safe export는 "
                "mutually-exclusive class labels만 지원합니다."
            )

        num_heads = int(
            prediction_logits.shape[0]
        )

        expected_labels = [
            0,
            *[
                int(value)
                for value
                in self.label_manager.foreground_labels
            ],
        ]

        if expected_labels != list(
            range(num_heads)
        ):
            raise RuntimeError(
                "A-N00 Location label/channel mapping mismatch. "
                f"labels={expected_labels[:10]}... "
                f"num_heads={num_heads}"
            )

        if num_heads != 53:
            raise RuntimeError(
                "Dataset501 Location은 53 output heads가 "
                f"필요합니다. Actual={num_heads}"
            )

        # -----------------------------------------------------
        # Official nnU-Net export와 동일 target shape / spacing
        # -----------------------------------------------------

        target_shape = tuple(
            int(value)
            for value
            in properties[
                "shape_after_cropping_and_before_resampling"
            ]
        )

        spacing_transposed = [
            properties["spacing"][i]
            for i
            in self.plans_manager.transpose_forward
        ]

        if (
            len(
                self.configuration_manager.spacing
            )
            ==
            len(
                properties[
                    "shape_after_cropping_and_before_resampling"
                ]
            )
        ):
            current_spacing = (
                self.configuration_manager.spacing
            )

        else:
            current_spacing = [
                spacing_transposed[0],
                *self.configuration_manager.spacing,
            ]

        target_spacing = (
            spacing_transposed
        )

        seg_dtype = (
            np.uint8
            if len(
                self.label_manager.foreground_labels
            ) < 255
            else np.uint16
        )

        segmentation = np.zeros(
            target_shape,
            dtype=seg_dtype,
        )

        best_score = None

        # -----------------------------------------------------
        # Class-by-class resampling
        #
        # np.argmax와 동일하게 동점일 경우
        # 먼저 등장한 class를 유지하기 위해
        # 비교는 반드시 ">"를 사용한다.
        # -----------------------------------------------------

        for class_index in range(
            num_heads
        ):

            channel = (
                prediction_logits[
                    class_index:
                    class_index + 1
                ]
            )

            if isinstance(
                channel,
                torch.Tensor,
            ):

                channel_numpy = (
                    channel
                    .detach()
                    .to(
                        device="cpu"
                    )
                    .numpy()
                )

            else:

                channel_numpy = (
                    np.asarray(
                        channel
                    )
                )

            resampled = (
                self.configuration_manager
                .resampling_fn_probabilities(
                    channel_numpy,
                    target_shape,
                    current_spacing,
                    target_spacing,
                )
            )

            if isinstance(
                resampled,
                torch.Tensor,
            ):

                score = (
                    resampled[0]
                    .detach()
                    .cpu()
                    .numpy()
                )

            else:

                score = np.asarray(
                    resampled[0]
                )

            if best_score is None:

                best_score = np.array(
                    score,
                    copy=True,
                )

            else:

                better = (
                    score
                    > best_score
                )

                segmentation[
                    better
                ] = class_index

                best_score[
                    better
                ] = score[
                    better
                ]

                del better

            del channel
            del channel_numpy
            del resampled
            del score

            gc.collect()

        if best_score is None:
            raise RuntimeError(
                "Final Validation argmax 생성 실패"
            )

        del best_score

        # -----------------------------------------------------
        # Revert cropping
        # -----------------------------------------------------

        shape_before_cropping = tuple(
            int(value)
            for value
            in properties[
                "shape_before_cropping"
            ]
        )

        segmentation_reverted = np.zeros(
            shape_before_cropping,
            dtype=seg_dtype,
        )

        bbox = properties[
            "bbox_used_for_cropping"
        ]

        bbox_slicer = tuple(
            slice(
                int(bounds[0]),
                int(bounds[1]),
            )
            for bounds
            in bbox
        )

        expected_crop_shape = tuple(
            int(bounds[1]) - int(bounds[0])
            for bounds
            in bbox
        )

        if (
            tuple(segmentation.shape)
            != expected_crop_shape
        ):

            raise RuntimeError(
                "Final segmentation crop shape mismatch: "
                f"{tuple(segmentation.shape)} "
                f"!= {expected_crop_shape}"
            )

        segmentation_reverted[
            bbox_slicer
        ] = segmentation

        del segmentation

        # -----------------------------------------------------
        # Revert transpose
        # -----------------------------------------------------

        segmentation_reverted = (
            segmentation_reverted
            .transpose(
                self.plans_manager.transpose_backward
            )
        )

        # -----------------------------------------------------
        # Original geometry로 저장
        # -----------------------------------------------------

        image_reader_writer = (
            self.plans_manager
            .image_reader_writer_class()
        )

        output_file = (
            str(
                output_filename_truncated
            )
            +
            self.dataset_json[
                "file_ending"
            ]
        )

        image_reader_writer.write_seg(
            segmentation_reverted,
            output_file,
            properties,
        )

        del segmentation_reverted

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def _run_patient_full_volume_validation(
        self,
    ):
        """
        Perform full-volume sliding-window inference on every
        validation patient.

        Predictions are exported to a temporary local directory
        only long enough to restore original image geometry and
        calculate metrics. They are not accumulated across epochs.
        """

        if self.is_ddp:
            raise RuntimeError(
                "A-N00 patient-level validation currently expects "
                "single-GPU training. Current run is DDP."
            )

        _, val_keys = (
            self.do_split()
        )

        val_keys = list(
            val_keys
        )

        if (
            len(val_keys)
            != self.expected_validation_cases
        ):
            raise RuntimeError(
                "Validation case count mismatch. "
                f"Expected={self.expected_validation_cases}, "
                f"Actual={len(val_keys)}"
            )

        dataset_val = self.dataset_class(
            self.preprocessed_dataset_folder,
            val_keys,
            folder_with_segs_from_previous_stage=(
                self.folder_with_segs_from_previous_stage
            ),
        )

        if self.label_manager.has_regions:
            raise RuntimeError(
                "A-N00 Location validation expects "
                "standard mutually-exclusive class labels, "
                "not region-based labels."
            )

        foreground_labels = [
            int(label)
            for label
            in self.label_manager.foreground_labels
        ]

        if not foreground_labels:
            raise RuntimeError(
                "No foreground labels found."
            )

        label_name_by_id = {}

        for name, value in (
            self.dataset_json
            .get(
                "labels",
                {}
            )
            .items()
        ):

            if isinstance(
                value,
                int,
            ):
                label_name_by_id[
                    int(value)
                ] = name

        if (
            self._final_selected_best_epoch_1based
            is None
        ):
            raise RuntimeError(
                "Final Validation 시작 전 "
                "selected best epoch이 설정되지 않았습니다."
            )

        selected_best_epoch_1based = int(
            self._final_selected_best_epoch_1based
        )

        class_counts = {
            label: {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "gt_voxels": 0,
                "pred_voxels": 0,
            }
            for label
            in foreground_labels
        }

        # =====================================================
        # FINAL PATIENT VALIDATION RESUME
        # =====================================================

        final_resume_state = (
            self._load_final_patient_resume_state()
        )

        if final_resume_state is None:

            patient_rows = []

            negative_cases = 0
            negative_case_fp = 0

            completed_case_ids = set()

            resumed_patient_count = 0

        else:

            patient_rows = list(
                final_resume_state[
                    "patient_rows"
                ]
            )

            class_counts = (
                final_resume_state[
                    "class_counts"
                ]
            )

            negative_cases = int(
                final_resume_state[
                    "negative_cases"
                ]
            )

            negative_case_fp = int(
                final_resume_state[
                    "negative_case_fp"
                ]
            )

            completed_case_ids = set(
                final_resume_state[
                    "completed_case_ids"
                ]
            )

            resumed_patient_count = len(
                completed_case_ids
            )

            self.print_to_log_file(
                "[FINAL-PATIENT-VAL RESUME] "
                f"completed="
                f"{resumed_patient_count}/"
                f"{len(val_keys)} | "
                f"next patient="
                f"{resumed_patient_count + 1}"
            )

        self.set_deep_supervision_enabled(
            False
        )

        self.network.eval()

        # =====================================================
        # A_N00_FINAL_MEMORY_PRE_CLEANUP_V2
        # =====================================================

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,

            # Safer on a T4 with 53-class Location output:
            # network inference remains on GPU while accumulation
            # can fall back to CPU memory.
            perform_everything_on_device=True,

            device=self.device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )

        predictor.manual_initialization(
            self.network,
            self.plans_manager,
            self.configuration_manager,
            None,
            self.dataset_json,
            self.__class__.__name__,
            self.inference_allowed_mirroring_axes,
        )

        image_reader_writer = (
            self.plans_manager
            .image_reader_writer_class()
        )

        file_ending = (
            self.dataset_json[
                "file_ending"
            ]
        )

        gt_root = (
            Path(
                self.preprocessed_dataset_folder_base
            )
            / "gt_segmentations"
        )

        epoch_start = (
            time.perf_counter()
        )

        try:

            with tempfile.TemporaryDirectory(
                prefix=(
                    f"a_n00_val_epoch_"
                    f"{self.current_epoch + 1:03d}_"
                )
            ) as tmp_dir_string:

                tmp_dir = Path(
                    tmp_dir_string
                )

                validation_identifiers = list(
                    dataset_val.identifiers
                )

                for patient_index, case_id in enumerate(
                    validation_identifiers,
                    start=1,
                ):

                    if (
                        case_id
                        in completed_case_ids
                    ):
                        continue

                    patient_start = (
                        time.perf_counter()
                    )

                    data, _, seg_prev, properties = (
                        dataset_val.load_case(
                            case_id
                        )
                    )

                    # Blosc2 -> NumPy
                    data = data[:]

                    if self.is_cascaded:

                        seg_prev = (
                            seg_prev[:]
                        )

                        data = np.vstack(
                            (
                                data,
                                convert_labelmap_to_one_hot(
                                    seg_prev,
                                    self.label_manager.foreground_labels,
                                    output_dtype=data.dtype,
                                ),
                            )
                        )

                    with warnings.catch_warnings():

                        warnings.simplefilter(
                            "ignore"
                        )

                        data_tensor = (
                            torch.from_numpy(
                                data
                            )
                        )

                    with torch.no_grad():

                        prediction_logits = (
                            predictor
                            .predict_sliding_window_return_logits(
                                data_tensor
                            )
                        )

                    output_filename_truncated = str(
                        tmp_dir
                        / case_id
                    )

                    # Same original-geometry restoration pipeline
                    # used by official nnU-Net validation.
                    self._export_prediction_memory_safe(
                        prediction_logits,
                        properties,
                        output_filename_truncated,
                    )

                    del prediction_logits

                    gc.collect()

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    pred_file = Path(
                        f"{output_filename_truncated}"
                        f"{file_ending}"
                    )

                    gt_file = (
                        gt_root
                        / f"{case_id}"
                          f"{file_ending}"
                    )

                    if not pred_file.is_file():
                        raise FileNotFoundError(
                            f"Prediction export missing: "
                            f"{pred_file}"
                        )

                    if not gt_file.is_file():
                        raise FileNotFoundError(
                            f"Validation GT missing: "
                            f"{gt_file}"
                        )

                    pred_array, _ = (
                        image_reader_writer
                        .read_seg(
                            str(
                                pred_file
                            )
                        )
                    )

                    gt_array, _ = (
                        image_reader_writer
                        .read_seg(
                            str(
                                gt_file
                            )
                        )
                    )

                    pred_array = np.asarray(
                        pred_array
                    )

                    gt_array = np.asarray(
                        gt_array
                    )

                    # ReaderWriter returns channel-first segmentation.
                    if (
                        pred_array.ndim >= 4
                        and pred_array.shape[0] == 1
                    ):
                        pred_array = (
                            pred_array[0]
                        )

                    if (
                        gt_array.ndim >= 4
                        and gt_array.shape[0] == 1
                    ):
                        gt_array = (
                            gt_array[0]
                        )

                    if (
                        pred_array.shape
                        != gt_array.shape
                    ):
                        raise RuntimeError(
                            f"{case_id}: prediction/GT shape mismatch "
                            f"{pred_array.shape} != {gt_array.shape}"
                        )

                    patient_metric = (
                        self._patient_binary_metrics(
                            gt_array,
                            pred_array,
                        )
                    )

                    if (
                        patient_metric[
                            "is_negative_case"
                        ]
                    ):
                        negative_cases += 1

                        negative_case_fp += (
                            patient_metric[
                                "false_positive_on_negative"
                            ]
                        )

                    # -------------------------------------------------
                    # Per-patient Location Macro Dice
                    #
                    # 한 환자 GT에 실제 존재하는 Location class만
                    # 동일 가중치로 평균한다.
                    # -------------------------------------------------

                    patient_macro_dices = []
                    patient_macro_gt_present_classes = 0

                    for label in foreground_labels:

                        gt_mask = (
                            gt_array
                            == label
                        )

                        pred_mask = (
                            pred_array
                            == label
                        )

                        tp = int(
                            np.count_nonzero(
                                gt_mask
                                & pred_mask
                            )
                        )

                        fp = int(
                            np.count_nonzero(
                                (~gt_mask)
                                & pred_mask
                            )
                        )

                        fn = int(
                            np.count_nonzero(
                                gt_mask
                                & (~pred_mask)
                            )
                        )

                        gt_voxels = int(
                            np.count_nonzero(
                                gt_mask
                            )
                        )

                        pred_voxels = int(
                            np.count_nonzero(
                                pred_mask
                            )
                        )

                        # ---------------------------------------------
                        # Patient Macro Dice contribution
                        #
                        # 이 환자의 GT에 존재하는 class만 포함.
                        # GT class가 존재하지만 prediction이 없으면
                        # 해당 class Dice는 0.
                        # ---------------------------------------------

                        if gt_voxels > 0:

                            patient_macro_gt_present_classes += 1

                            patient_class_dice_denominator = (
                                2 * tp
                                + fp
                                + fn
                            )

                            patient_class_dice = (
                                float(
                                    2 * tp
                                    / patient_class_dice_denominator
                                )
                                if patient_class_dice_denominator > 0
                                else 0.0
                            )

                            patient_macro_dices.append(
                                patient_class_dice
                            )

                        class_counts[
                            label
                        ][
                            "tp"
                        ] += tp

                        class_counts[
                            label
                        ][
                            "fp"
                        ] += fp

                        class_counts[
                            label
                        ][
                            "fn"
                        ] += fn

                        class_counts[
                            label
                        ][
                            "gt_voxels"
                        ] += gt_voxels

                        class_counts[
                            label
                        ][
                            "pred_voxels"
                        ] += pred_voxels

                    patient_macro_dice = (
                        self._finite_mean(
                            patient_macro_dices
                        )
                    )

                    patient_row = {
                        "epoch_0based":
                            selected_best_epoch_1based - 1,

                        "epoch_1based":
                            selected_best_epoch_1based,

                        "case_id":
                            case_id,

                        # Foreground-merged patient metric
                        **patient_metric,

                        # Location class-wise patient Macro Dice
                        "patient_macro_dice":
                            patient_macro_dice,

                        "patient_macro_gt_present_classes":
                            patient_macro_gt_present_classes,
                    }

                    patient_rows.append(
                        patient_row
                    )

                    completed_case_ids.add(
                        case_id
                    )

                    # -----------------------------------------
                    # Persist immediately after this patient.
                    #
                    # If runtime dies during NEXT patient,
                    # this completed patient is never repeated.
                    # -----------------------------------------

                    self._save_final_patient_resume_state(
                        patient_rows=patient_rows,
                        class_counts=class_counts,
                        negative_cases=negative_cases,
                        negative_case_fp=negative_case_fp,
                        completed_case_ids=completed_case_ids,
                    )

                    elapsed = (
                        time.perf_counter()
                        - epoch_start
                    )

                    completed_total = len(
                        patient_rows
                    )

                    completed_this_session = max(
                        1,
                        completed_total
                        - resumed_patient_count,
                    )

                    avg_patient_seconds = (
                        elapsed
                        / completed_this_session
                    )

                    remaining = (
                        len(val_keys)
                        - completed_total
                    )

                    eta_seconds = (
                        avg_patient_seconds
                        * remaining
                    )

                    if (
                        completed_total
                        % self.patient_progress_every
                        == 0

                        or completed_total
                        == len(val_keys)
                    ):

                        running_mean_dice = (
                            self._finite_mean(
                                [
                                    row[
                                        "dice"
                                    ]
                                    for row
                                    in patient_rows
                                ]
                            )
                        )

                        pct = (
                            100.0
                            * completed_total
                            / len(val_keys)
                        )

                        print(
                            "[FINAL-PATIENT-VAL] "
                            f"{completed_total}/"
                            f"{len(val_keys)} "
                            f"({pct:.1f}%) | "
                            f"case={case_id} | "
                            f"dice="
                            f"{self._format_metric(patient_metric['dice'])} | "
                            f"patient_macro="
                            f"{self._format_metric(patient_macro_dice)} | "
                            f"running_mean="
                            f"{self._format_metric(running_mean_dice)} | "
                            f"ETA={eta_seconds / 60.0:.1f} min",
                            flush=True,
                        )

                    del (
                        prediction_logits,
                        data_tensor,
                        data,
                        pred_array,
                        gt_array,
                    )

                    # Prediction is no longer needed after metrics.
                    try:
                        pred_file.unlink()
                    except FileNotFoundError:
                        pass

        finally:

            self.set_deep_supervision_enabled(
                True
            )

            compute_gaussian.cache_clear()

            empty_cache(
                self.device
            )

        # =====================================================
        # PATIENT-LEVEL SUMMARY
        # =====================================================

        patient_mean_dice = (
            self._finite_mean(
                [
                    row["dice"]
                    for row
                    in patient_rows
                ]
            )
        )

        patient_median_dice = (
            self._finite_median(
                [
                    row["dice"]
                    for row
                    in patient_rows
                ]
            )
        )

        # 46명 각각의 Patient Macro Dice를 다시 평균한
        # Validation Patient Mean Macro Dice
        patient_mean_macro_dice = (
            self._finite_mean(
                [
                    row["patient_macro_dice"]
                    for row
                    in patient_rows
                ]
            )
        )

        patient_mean_precision = (
            self._finite_mean(
                [
                    row["precision"]
                    for row
                    in patient_rows
                ]
            )
        )

        patient_mean_recall = (
            self._finite_mean(
                [
                    row["recall"]
                    for row
                    in patient_rows
                ]
            )
        )

        negative_case_fp_rate = (
            float(
                negative_case_fp
                / negative_cases
            )
            if negative_cases > 0
            else float("nan")
        )

        # =====================================================
        # MACRO CLASS METRICS
        #
        # Only classes with GT present anywhere in validation
        # are included in the macro average.
        # =====================================================

        class_rows = []

        macro_dices = []
        macro_precisions = []
        macro_recalls = []

        for label in foreground_labels:

            counts = (
                class_counts[
                    label
                ]
            )

            tp = counts[
                "tp"
            ]

            fp = counts[
                "fp"
            ]

            fn = counts[
                "fn"
            ]

            gt_voxels = counts[
                "gt_voxels"
            ]

            pred_voxels = counts[
                "pred_voxels"
            ]

            gt_present = (
                gt_voxels > 0
            )

            if gt_present:

                dice_denominator = (
                    2 * tp
                    + fp
                    + fn
                )

                dice = (
                    float(
                        2 * tp
                        / dice_denominator
                    )
                    if dice_denominator > 0
                    else 0.0
                )

                precision_denominator = (
                    tp + fp
                )

                precision = (
                    float(
                        tp
                        / precision_denominator
                    )
                    if precision_denominator > 0
                    else 0.0
                )

                recall_denominator = (
                    tp + fn
                )

                recall = (
                    float(
                        tp
                        / recall_denominator
                    )
                    if recall_denominator > 0
                    else 0.0
                )

                macro_dices.append(
                    dice
                )

                macro_precisions.append(
                    precision
                )

                macro_recalls.append(
                    recall
                )

            else:

                dice = float(
                    "nan"
                )

                precision = float(
                    "nan"
                )

                recall = float(
                    "nan"
                )

            class_rows.append(
                {
                    "epoch_0based":
                        self.current_epoch,

                    "epoch_1based":
                        self.current_epoch + 1,

                    "class_id":
                        label,

                    "class_name":
                        label_name_by_id.get(
                            label,
                            f"class_{label}",
                        ),

                    "gt_present":
                        int(
                            gt_present
                        ),

                    "gt_voxels":
                        gt_voxels,

                    "pred_voxels":
                        pred_voxels,

                    "tp":
                        tp,

                    "fp":
                        fp,

                    "fn":
                        fn,

                    "dice":
                        dice,

                    "precision":
                        precision,

                    "recall":
                        recall,
                }
            )

        macro_dice = (
            self._finite_mean(
                macro_dices
            )
        )

        macro_precision = (
            self._finite_mean(
                macro_precisions
            )
        )

        macro_recall = (
            self._finite_mean(
                macro_recalls
            )
        )

        metrics = {
            "val_patient_mean_dice":
                patient_mean_dice,

            "val_patient_median_dice":
                patient_median_dice,

            "val_patient_mean_macro_dice":
                patient_mean_macro_dice,

            "val_patient_mean_precision":
                patient_mean_precision,

            "val_patient_mean_recall":
                patient_mean_recall,

            "val_macro_dice":
                macro_dice,

            "val_macro_precision":
                macro_precision,

            "val_macro_recall":
                macro_recall,

            "negative_case_fp_rate":
                negative_case_fp_rate,

            "validation_cases":
                len(
                    patient_rows
                ),

            "negative_cases":
                negative_cases,

            "negative_cases_with_fp":
                negative_case_fp,

            "macro_gt_present_classes":
                len(
                    macro_dices
                ),
        }

        self._latest_patient_metrics = (
            metrics
        )

        self._latest_patient_rows = (
            patient_rows
        )

        self._latest_class_rows = (
            class_rows
        )

        self.print_to_log_file(
            "\n[FINAL-PATIENT-VAL SUMMARY] "
            f"BestEpoch={selected_best_epoch_1based} | "
            f"PatientMeanDice="
            f"{self._format_metric(patient_mean_dice)} | "
            f"PatientMedianDice="
            f"{self._format_metric(patient_median_dice)} | "
            f"PatientMeanMacroDice="
            f"{self._format_metric(patient_mean_macro_dice)} | "
            f"SplitMacroDice="
            f"{self._format_metric(macro_dice)} | "
            f"Precision="
            f"{self._format_metric(patient_mean_precision)} | "
            f"Recall="
            f"{self._format_metric(patient_mean_recall)} | "
            f"NegativeFP="
            f"{self._format_metric(negative_case_fp_rate)}"
        )

        return (
            metrics,
            patient_rows,
            class_rows,
        )

    # =========================================================
    # CUSTOM EPOCH END
    #
    # Replaces nnU-Net default EMA pseudo-Dice best checkpoint
    # with patient-level full-volume Mean Dice.
    # =========================================================

    def on_epoch_end(
        self,
    ):

        self.logger.log(
            "epoch_end_timestamps",
            time.time(),
            self.current_epoch,
        )

        train_loss = float(
            self.logger.get_value(
                "train_losses",
                step=-1,
            )
        )

        val_loss = float(
            self.logger.get_value(
                "val_losses",
                step=-1,
            )
        )

        online_dice = float(
            self.logger.get_value(
                "mean_fg_dice",
                step=-1,
            )
        )

        ema_online_dice = float(
            self.logger.get_value(
                "ema_fg_dice",
                step=-1,
            )
        )


        if not np.isfinite(
            ema_online_dice
        ):

            raise RuntimeError(
                "ema_fg_dice is not finite. "
                "Best checkpoint cannot be selected."
            )


        self.print_to_log_file(
            "train_loss",
            np.round(
                train_loss,
                decimals=4,
            ),
        )

        self.print_to_log_file(
            "online_val_loss",
            np.round(
                val_loss,
                decimals=4,
            ),
        )

        self.print_to_log_file(
            "online_patch_mean_fg_dice",
            np.round(
                online_dice,
                decimals=4,
            ),
        )

        self.print_to_log_file(
            "online_patch_ema_fg_dice",
            np.round(
                ema_online_dice,
                decimals=4,
            ),
            "(Best / EarlyStopping monitor)",
        )


        # =====================================================
        # BEST CHECKPOINT = TRUE MAX ema_fg_dice
        #
        # min_delta is ONLY for EarlyStopping reset.
        # =====================================================

        previous_best = (
            self._best_ema
        )

        is_new_best = (
            previous_best is None
            or ema_online_dice
            > float(
                previous_best
            )
        )

        if is_new_best:

            self._best_ema = (
                ema_online_dice
            )

            self.print_to_log_file(
                "[BEST] New best "
                f"ema_fg_dice="
                f"{ema_online_dice:.6f}"
            )

            self.save_checkpoint(
                str(
                    Path(
                        self.output_folder
                    )
                    / "checkpoint_best.pth"
                )
            )


        epoch_seconds = float(
            self.logger.get_value(
                "epoch_end_timestamps",
                step=-1,
            )
            - self.logger.get_value(
                "epoch_start_timestamps",
                step=-1,
            )
        )

        self.print_to_log_file(
            f"Epoch time: "
            f"{epoch_seconds:.2f} s"
        )


        # =====================================================
        # TRAINING HISTORY
        #
        # There are intentionally NO Patient Full-volume
        # metrics in per-epoch history anymore.
        # =====================================================

        history_row = {
            "epoch_0based":
                int(
                    self.current_epoch
                ),

            "epoch_1based":
                int(
                    self.current_epoch + 1
                ),

            "monitor":
                "ema_fg_dice",

            "train_loss":
                train_loss,

            "online_val_loss":
                val_loss,

            "online_patch_mean_fg_dice":
                online_dice,

            "online_patch_ema_fg_dice":
                ema_online_dice,

            "best_ema_fg_dice":
                float(
                    self._best_ema
                ),

            "is_best":
                int(
                    is_new_best
                ),

            "epoch_seconds":
                epoch_seconds,
        }

        self._upsert_training_history(
            history_row
        )


        # =====================================================
        # EPOCH-COMPLETE CHECKPOINT
        #
        # Always write latest, INCLUDING the final epoch.
        #
        # Official save_checkpoint stores current_epoch + 1,
        # which is exactly what we want after a fully completed
        # epoch.
        # =====================================================

        self.save_checkpoint(
            str(
                Path(
                    self.output_folder
                )
                / "checkpoint_latest.pth"
            )
        )


        if self.local_rank == 0:

            self.logger.plot_progress_png(
                self.output_folder
            )


        self.current_epoch += 1


    # =========================================================
    # TRAINING LOOP
    # =========================================================

    def run_training(
        self,
    ):

        # =====================================================
        # INITIALIZE
        # =====================================================

        self.on_train_start()

        # get_dataloaders() prefetches one train + one val batch.
        # Undo those RNG effects when --c restored a strict
        # mid-epoch checkpoint.
        self._restore_pending_rng_state()

        self._restore_ema_best_from_history()


        self.print_to_log_file(
            "\n"
            + "=" * 62
        )

        self.print_to_log_file(
            "A-N00 FINAL TRAINING POLICY V2"
        )

        self.print_to_log_file(
            f"Max epochs           : "
            f"{self.num_epochs}"
        )

        self.print_to_log_file(
            f"Train iterations      : "
            f"{self.num_iterations_per_epoch}"
        )

        self.print_to_log_file(
            f"Online val iterations : "
            f"{self.num_val_iterations_per_epoch}"
        )

        self.print_to_log_file(
            "Best monitor          : "
            "ema_fg_dice"
        )

        self.print_to_log_file(
            "EarlyStopping monitor : "
            "ema_fg_dice"
        )

        self.print_to_log_file(
            f"Patience              : "
            f"{self.early_stopping_patience}"
        )

        self.print_to_log_file(
            f"Min delta             : "
            f"{self.early_stopping_min_delta}"
        )

        self.print_to_log_file(
            "Full-volume per epoch : DISABLED"
        )

        self.print_to_log_file(
            "Final Full-volume Val : "
            "46 patients ONCE after training"
        )

        self.print_to_log_file(
            "Final Val checkpoint  : "
            "checkpoint_best.pth"
        )

        self.print_to_log_file(
            "Train Resume          : "
            "1 completed iteration"
        )

        self.print_to_log_file(
            "Online-Val Resume     : "
            "1 completed iteration"
        )

        self.print_to_log_file(
            "Final Patient Resume  : "
            "1 completed patient"
        )

        self.print_to_log_file(
            "=" * 62
            + "\n"
        )


        # =====================================================
        # Was EarlyStopping already reached before a resumed
        # final-validation session?
        # =====================================================

        early_stop_already_reached = False

        if (
            self.current_epoch > 0
            and self._monitor_history()
        ):

            _, bad_epochs = (
                self._calculate_early_stopping_state()
            )

            early_stop_already_reached = (
                bad_epochs
                >= self.early_stopping_patience
            )


        # =====================================================
        # TRAINING LOOP
        # =====================================================

        if not early_stop_already_reached:

            while (
                self.current_epoch
                < self.num_epochs
            ):

                phase = (
                    self._resume_phase
                )

                if phase not in {
                    "epoch_start",
                    "train",
                    "online_val",
                    "epoch_end",
                }:

                    raise RuntimeError(
                        f"Invalid resume phase: {phase}"
                    )


                # =============================================
                # FRESH EPOCH
                # =============================================

                if phase == "epoch_start":

                    self.on_epoch_start()

                    self.on_train_epoch_start()

                    train_start_iteration = 0

                    train_outputs = []

                    phase = "train"


                # =============================================
                # RESUME TRAIN
                # =============================================

                elif phase == "train":

                    train_start_iteration = int(
                        self._resume_train_next_iteration
                    )

                    train_outputs = [
                        {
                            "loss":
                                np.asarray(
                                    value,
                                    dtype=np.float32,
                                )
                        }
                        for value
                        in self._resume_train_losses
                    ]

                    if (
                        len(
                            train_outputs
                        )
                        != train_start_iteration
                    ):

                        raise RuntimeError(
                            "Strict resume train loss count "
                            "mismatch: "
                            f"losses="
                            f"{len(train_outputs)}, "
                            f"completed="
                            f"{train_start_iteration}"
                        )

                    # Network mode is not checkpoint state.
                    # Set it without re-running LR scheduler.
                    self.network.train()

                    self.print_to_log_file(
                        "[TRAIN RESUME] "
                        f"Epoch "
                        f"{self.current_epoch + 1}/"
                        f"{self.num_epochs} | "
                        f"completed="
                        f"{train_start_iteration}/"
                        f"{self.num_iterations_per_epoch} | "
                        f"next="
                        f"{train_start_iteration + 1}/"
                        f"{self.num_iterations_per_epoch}"
                    )


                else:

                    train_start_iteration = (
                        self.num_iterations_per_epoch
                    )

                    train_outputs = []


                # =============================================
                # TRAIN ITERATIONS
                # =============================================

                if phase == "train":

                    train_session_start = (
                        time.perf_counter()
                    )

                    for batch_id in range(
                        train_start_iteration,
                        self.num_iterations_per_epoch,
                    ):

                        output = (
                            self.train_step(
                                next(
                                    self.dataloader_train
                                )
                            )
                        )

                        train_outputs.append(
                            output
                        )

                        completed_iteration = (
                            batch_id + 1
                        )

                        train_losses = [
                            float(
                                item[
                                    "loss"
                                ]
                            )
                            for item
                            in train_outputs
                        ]


                        # =====================================
                        # CRITICAL RESUME POINT
                        #
                        # Write AFTER optimizer step completes.
                        # =====================================

                        self._save_strict_resume_checkpoint(
                            phase="train",
                            train_next_iteration=(
                                completed_iteration
                            ),
                            train_losses=train_losses,
                        )


                        if (
                            completed_iteration
                            % self.progress_every
                            == 0

                            or completed_iteration
                            == self.num_iterations_per_epoch
                        ):

                            elapsed = (
                                time.perf_counter()
                                - train_session_start
                            )

                            this_session_done = max(
                                1,
                                completed_iteration
                                - train_start_iteration,
                            )

                            sec_per_iteration = (
                                elapsed
                                / this_session_done
                            )

                            remaining = (
                                self.num_iterations_per_epoch
                                - completed_iteration
                            )

                            eta_seconds = (
                                sec_per_iteration
                                * remaining
                            )

                            running_loss = float(
                                np.mean(
                                    train_losses
                                )
                            )

                            pct = (
                                100.0
                                * completed_iteration
                                / self.num_iterations_per_epoch
                            )

                            print(
                                "[TRAIN] "
                                f"Epoch "
                                f"{self.current_epoch + 1}/"
                                f"{self.num_epochs} | "
                                f"{completed_iteration}/"
                                f"{self.num_iterations_per_epoch} "
                                f"({pct:.1f}%) | "
                                f"loss="
                                f"{running_loss:.6f} | "
                                f"ETA="
                                f"{eta_seconds / 60.0:.1f} min | "
                                f"resume_saved="
                                f"{completed_iteration}",
                                flush=True,
                            )


                    self.on_train_epoch_end(
                        train_outputs
                    )


                    # Training for this epoch is complete.
                    # Save full checkpoint once more to mark
                    # transition into Online Validation.
                    self._save_strict_resume_checkpoint(
                        phase="online_val",
                        train_next_iteration=0,
                        train_losses=[],
                    )

                    phase = (
                        "online_val"
                    )


                # =============================================
                # ONLINE PATCH VALIDATION
                # =============================================

                if phase == "online_val":

                    self.on_validation_epoch_start()

                    online_state = (
                        self._load_online_val_resume_state()
                    )

                    if online_state is None:

                        val_outputs = []

                        val_start_iteration = 0

                    else:

                        val_outputs = list(
                            online_state[
                                "val_outputs"
                            ]
                        )

                        val_start_iteration = int(
                            online_state[
                                "next_iteration"
                            ]
                        )

                        # Sidecar RNG represents the point
                        # immediately after the last completed
                        # Online-Val iteration.
                        self._restore_rng_state(
                            online_state[
                                "rng_state"
                            ]
                        )

                        self.print_to_log_file(
                            "[ONLINE-VAL RESUME] "
                            f"Epoch "
                            f"{self.current_epoch + 1}/"
                            f"{self.num_epochs} | "
                            f"completed="
                            f"{val_start_iteration}/"
                            f"{self.num_val_iterations_per_epoch} | "
                            f"next="
                            f"{val_start_iteration + 1}/"
                            f"{self.num_val_iterations_per_epoch}"
                        )


                    val_session_start = (
                        time.perf_counter()
                    )


                    for batch_id in range(
                        val_start_iteration,
                        self.num_val_iterations_per_epoch,
                    ):

                        with torch.no_grad():

                            output = (
                                self.validation_step(
                                    next(
                                        self.dataloader_val
                                    )
                                )
                            )

                        val_outputs.append(
                            output
                        )

                        completed_iteration = (
                            batch_id + 1
                        )


                        # Small sidecar only.
                        # Model weights do not change in val.
                        self._save_online_val_resume_state(
                            next_iteration=(
                                completed_iteration
                            ),
                            val_outputs=val_outputs,
                        )


                        if (
                            completed_iteration
                            % self.progress_every
                            == 0

                            or completed_iteration
                            == self.num_val_iterations_per_epoch
                        ):

                            elapsed = (
                                time.perf_counter()
                                - val_session_start
                            )

                            this_session_done = max(
                                1,
                                completed_iteration
                                - val_start_iteration,
                            )

                            sec_per_iteration = (
                                elapsed
                                / this_session_done
                            )

                            remaining = (
                                self.num_val_iterations_per_epoch
                                - completed_iteration
                            )

                            eta_seconds = (
                                sec_per_iteration
                                * remaining
                            )

                            running_val_loss = float(
                                np.mean(
                                    [
                                        float(
                                            item[
                                                "loss"
                                            ]
                                        )
                                        for item
                                        in val_outputs
                                    ]
                                )
                            )

                            pct = (
                                100.0
                                * completed_iteration
                                / self.num_val_iterations_per_epoch
                            )

                            print(
                                "[ONLINE-VAL] "
                                f"Epoch "
                                f"{self.current_epoch + 1}/"
                                f"{self.num_epochs} | "
                                f"{completed_iteration}/"
                                f"{self.num_val_iterations_per_epoch} "
                                f"({pct:.1f}%) | "
                                f"loss="
                                f"{running_val_loss:.6f} | "
                                f"ETA="
                                f"{eta_seconds / 60.0:.1f} min | "
                                f"resume_saved="
                                f"{completed_iteration}",
                                flush=True,
                            )


                    self.on_validation_epoch_end(
                        val_outputs
                    )


                    # logger now contains:
                    # mean_fg_dice / ema_fg_dice / val_loss.
                    self._save_strict_resume_checkpoint(
                        phase="epoch_end",
                        train_next_iteration=0,
                        train_losses=[],
                    )

                    phase = (
                        "epoch_end"
                    )


                # =============================================
                # EPOCH END
                #
                # NO Full-volume Patient Validation here.
                # =============================================

                if phase == "epoch_end":

                    completed_epoch_0based = int(
                        self.current_epoch
                    )

                    self.on_epoch_end()

                    self._clear_online_val_resume_state(
                        completed_epoch_0based
                    )

                    self._resume_phase = (
                        "epoch_start"
                    )

                    self._resume_train_next_iteration = 0

                    self._resume_train_losses = []


                # =============================================
                # EARLY STOPPING
                # =============================================

                if self._should_early_stop():

                    self.print_to_log_file(
                        "\n"
                        + "=" * 62
                    )

                    self.print_to_log_file(
                        "EARLY STOPPING TRIGGERED"
                    )

                    self.print_to_log_file(
                        "monitor="
                        "ema_fg_dice"
                    )

                    self.print_to_log_file(
                        f"min_delta="
                        f"{self.early_stopping_min_delta}"
                    )

                    self.print_to_log_file(
                        f"no_improvement="
                        f"{self.early_stopping_patience}/"
                        f"{self.early_stopping_patience}"
                    )

                    self.print_to_log_file(
                        "=" * 62
                        + "\n"
                    )

                    break


        else:

            self.print_to_log_file(
                "[RESUME] EarlyStopping had already "
                "completed before this session."
            )


        # =====================================================
        # TRAINING IS NOW FINISHED
        # =====================================================

        best_checkpoint = (
            Path(
                self.output_folder
            )
            / "checkpoint_best.pth"
        )

        latest_checkpoint = (
            Path(
                self.output_folder
            )
            / "checkpoint_latest.pth"
        )

        final_checkpoint = (
            Path(
                self.output_folder
            )
            / "checkpoint_final.pth"
        )


        if not best_checkpoint.is_file():

            raise RuntimeError(
                "Training finished but checkpoint_best.pth "
                "does not exist."
            )


        if not latest_checkpoint.is_file():

            raise RuntimeError(
                "Training finished but checkpoint_latest.pth "
                "does not exist. "
                "Final validation cannot enter resumable mode."
            )


        # =====================================================
        # Preserve LAST TRAINING STATE as checkpoint_final.
        #
        # Do this BEFORE loading checkpoint_best for final
        # patient validation.
        # =====================================================

        final_tmp = Path(
            str(
                final_checkpoint
            )
            + ".tmp"
        )

        shutil.copy2(
            latest_checkpoint,
            final_tmp,
        )

        os.replace(
            final_tmp,
            final_checkpoint,
        )


        # =====================================================
        # Load BEST checkpoint for final validation.
        # =====================================================

        raw_best = torch.load(
            best_checkpoint,
            map_location="cpu",
            weights_only=False,
        )

        selected_best_epoch_1based = int(
            raw_best[
                "current_epoch"
            ]
        )

        selected_best_ema = (
            raw_best.get(
                "_best_ema"
            )
        )

        self._final_selected_best_epoch_1based = (
            selected_best_epoch_1based
        )

        self._final_selected_best_ema = (
            None
            if selected_best_ema is None
            else float(
                selected_best_ema
            )
        )


        self.print_to_log_file(
            "\n"
            + "=" * 62
        )

        self.print_to_log_file(
            "FINAL 46-PATIENT FULL-VOLUME VALIDATION"
        )

        self.print_to_log_file(
            "Checkpoint : checkpoint_best.pth"
        )

        self.print_to_log_file(
            f"Best epoch : "
            f"{selected_best_epoch_1based}"
        )

        self.print_to_log_file(
            f"Best EMA   : "
            f"{self._format_metric(selected_best_ema)}"
        )

        self.print_to_log_file(
            "Cases      : "
            f"{self.expected_validation_cases}"
        )

        self.print_to_log_file(
            "Run count  : ONCE after training"
        )

        self.print_to_log_file(
            "=" * 62
            + "\n"
        )


        # Loading best changes only in-memory trainer state.
        # checkpoint_final already preserves last training state.
        self.load_checkpoint(
            str(
                best_checkpoint
            )
        )


        # load_checkpoint resets current epoch according to
        # best checkpoint; restore explicit metadata for output.
        self._final_selected_best_epoch_1based = (
            selected_best_epoch_1based
        )

        self._final_selected_best_ema = (
            None
            if selected_best_ema is None
            else float(
                selected_best_ema
            )
        )


        # =====================================================
        # ONE AND ONLY ONE FULL-VOLUME VALIDATION
        #
        # This function itself resumes from the next unfinished
        # patient if final_validation/resume_state.pth exists.
        # =====================================================

        (
            final_metrics,
            final_patient_rows,
            final_class_rows,
        ) = (
            self._run_patient_full_volume_validation()
        )


        if (
            len(
                final_patient_rows
            )
            != self.expected_validation_cases
        ):

            raise RuntimeError(
                "Final Validation patient count mismatch: "
                f"expected="
                f"{self.expected_validation_cases}, "
                f"actual="
                f"{len(final_patient_rows)}"
            )


        self._write_final_validation_outputs(
            metrics=final_metrics,
            patient_rows=final_patient_rows,
            class_rows=final_class_rows,
        )


        # COMPLETE.json already exists at this point.
        self._clear_final_patient_resume_state()


        # =====================================================
        # Only AFTER final validation is completely persisted
        # may latest checkpoint be removed.
        # =====================================================

        try:

            latest_checkpoint.unlink()

        except FileNotFoundError:

            pass


        # SingleThreadedAugmenter has no background worker
        # processes requiring _finish().
        self.dataloader_train = None
        self.dataloader_val = None

        empty_cache(
            self.device
        )


        self.print_to_log_file(
            "\n"
            + "=" * 62
        )

        self.print_to_log_file(
            "A-N00 TRAINING + FINAL VALIDATION COMPLETE"
        )

        self.print_to_log_file(
            "Final Validation outputs:"
        )

        self.print_to_log_file(
            str(
                self.final_validation_root
            )
        )

        self.print_to_log_file(
            "=" * 62
            + "\n"
        )

