
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerAneurysmEarlyStopping(nnUNetTrainer):
    """
    Standard nnU-Net architecture를 그대로 사용하는 A-N00 Trainer.

    추가 기능
    ----------
    1. max epoch 설정
    2. EarlyStopping
    3. 매 epoch checkpoint 저장 가능
    4. Train / Validation 진행률 실시간 출력
    5. checkpoint 저장 시 임시 파일을 사용하여 기존 checkpoint 보호
    6. nnU-Net 공식 --c Resume와 호환

    네트워크 구조, Loss, Optimizer 종류, Augmentation,
    Deep Supervision 등 Standard nnU-Net 기본 구조는 변경하지 않는다.
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
            os.environ.get("A_N00_MAX_EPOCHS", "300")
        )

        self.initial_lr = float(
            os.environ.get("A_N00_INITIAL_LR", "0.01")
        )

        self.weight_decay = float(
            os.environ.get("A_N00_WEIGHT_DECAY", "0.00003")
        )

        self.oversample_foreground_percent = float(
            os.environ.get("A_N00_OVERSAMPLE", "0.33")
        )

        self.num_iterations_per_epoch = int(
            os.environ.get("A_N00_TRAIN_ITERS", "250")
        )

        self.num_val_iterations_per_epoch = int(
            os.environ.get("A_N00_VAL_ITERS", "50")
        )

        # =====================================================
        # CHECKPOINT / PROGRESS
        # =====================================================

        # nnU-Net 기본은 50.
        # 본 프로젝트는 Runtime 중단 대응을 위해 기본값 1.
        self.save_every = int(
            os.environ.get("A_N00_SAVE_EVERY", "1")
        )

        # Train/Val 몇 iteration마다 진행률을 출력할지
        self.progress_every = max(
            1,
            int(os.environ.get("A_N00_PROGRESS_EVERY", "25"))
        )

        # =====================================================
        # EARLY STOPPING
        # =====================================================

        self.early_stopping_patience = int(
            os.environ.get("A_N00_PATIENCE", "30")
        )

        self.early_stopping_min_delta = float(
            os.environ.get("A_N00_MIN_DELTA", "0.0001")
        )

        self.early_stopping_monitor = os.environ.get(
            "A_N00_MONITOR",
            "ema_fg_dice"
        )

        self.early_stopping_start_epoch = int(
            os.environ.get("A_N00_EARLY_STOP_START", "0")
        )

        if self.early_stopping_monitor not in {
            "ema_fg_dice",
            "val_loss",
        }:
            raise ValueError(
                "A_N00_MONITOR must be "
                "'ema_fg_dice' or 'val_loss'."
            )


    # =========================================================
    # ATOMIC CHECKPOINT
    # =========================================================

    def save_checkpoint(self, filename: str) -> None:
        """
        nnU-Net 공식 checkpoint 형식은 그대로 유지한다.

        먼저 filename.tmp에 완전히 저장한 뒤
        os.replace로 실제 checkpoint를 교체한다.

        Runtime이 checkpoint 작성 중 종료되더라도
        이전 정상 checkpoint를 최대한 보존하기 위한 처리.
        """

        tmp_filename = f"{filename}.tmp"

        super().save_checkpoint(tmp_filename)

        if self.local_rank == 0:
            tmp_path = Path(tmp_filename)

            if tmp_path.exists():
                os.replace(tmp_filename, filename)


    # =========================================================
    # EARLY STOPPING STATE
    # =========================================================

    def _monitor_history(self):
        """
        nnU-Net checkpoint에는 logger history가 저장된다.
        따라서 별도의 EarlyStopping state를 checkpoint에
        추가하지 않고 history에서 상태를 재구성한다.

        이렇게 하면 nnU-Net 공식 --c Resume와
        checkpoint 형식을 그대로 유지할 수 있다.
        """

        if self.early_stopping_monitor == "ema_fg_dice":
            return self.logger.get_value(
                "ema_fg_dice",
                step=None
            )

        return self.logger.get_value(
            "val_losses",
            step=None
        )


    def _calculate_early_stopping_state(self):
        history = self._monitor_history()

        best = None
        bad_epochs = 0

        higher_is_better = (
            self.early_stopping_monitor == "ema_fg_dice"
        )

        for epoch, raw_score in enumerate(history):

            if epoch < self.early_stopping_start_epoch:
                continue

            if raw_score is None or not np.isfinite(raw_score):
                bad_epochs += 1
                continue

            score = float(raw_score)

            if best is None:
                best = score
                bad_epochs = 0
                continue

            if higher_is_better:
                improved = (
                    score >
                    best + self.early_stopping_min_delta
                )
            else:
                improved = (
                    score <
                    best - self.early_stopping_min_delta
                )

            if improved:
                best = score
                bad_epochs = 0
            else:
                bad_epochs += 1

        return best, bad_epochs


    def _should_early_stop(self):
        completed_epoch = self.current_epoch - 1

        if completed_epoch < self.early_stopping_start_epoch:
            return False

        best, bad_epochs = (
            self._calculate_early_stopping_state()
        )

        latest_history = self._monitor_history()

        latest = (
            latest_history[-1]
            if len(latest_history) > 0
            else None
        )

        self.print_to_log_file(
            "[EarlyStopping] "
            f"monitor={self.early_stopping_monitor} | "
            f"latest={latest} | "
            f"best={best} | "
            f"no_improvement={bad_epochs}/"
            f"{self.early_stopping_patience}"
        )

        return (
            bad_epochs
            >= self.early_stopping_patience
        )


    # =========================================================
    # TRAINING LOOP
    # =========================================================

    def run_training(self):

        self.on_train_start()

        self.print_to_log_file(
            "\n=============================================="
        )
        self.print_to_log_file(
            "A-N00 Standard nnU-Net Training Policy"
        )
        self.print_to_log_file(
            f"Max epochs       : {self.num_epochs}"
        )
        self.print_to_log_file(
            f"EarlyStop monitor: "
            f"{self.early_stopping_monitor}"
        )
        self.print_to_log_file(
            f"Patience         : "
            f"{self.early_stopping_patience}"
        )
        self.print_to_log_file(
            f"Min delta        : "
            f"{self.early_stopping_min_delta}"
        )
        self.print_to_log_file(
            f"EarlyStop start  : "
            f"{self.early_stopping_start_epoch}"
        )
        self.print_to_log_file(
            f"Save every       : {self.save_every} epoch"
        )
        self.print_to_log_file(
            f"Progress every   : "
            f"{self.progress_every} iteration"
        )
        self.print_to_log_file(
            "==============================================\n"
        )

        for epoch in range(
            self.current_epoch,
            self.num_epochs
        ):

            self.on_epoch_start()

            # =================================================
            # TRAIN
            # =================================================

            self.on_train_epoch_start()

            train_outputs = []

            for batch_id in range(
                self.num_iterations_per_epoch
            ):

                train_outputs.append(
                    self.train_step(
                        next(self.dataloader_train)
                    )
                )

                step = batch_id + 1

                if (
                    step % self.progress_every == 0
                    or
                    step == self.num_iterations_per_epoch
                ):
                    pct = (
                        100.0
                        * step
                        / self.num_iterations_per_epoch
                    )

                    print(
                        f"[TRAIN] "
                        f"Epoch {self.current_epoch} | "
                        f"{step}/"
                        f"{self.num_iterations_per_epoch} "
                        f"({pct:.1f}%)",
                        flush=True,
                    )

            self.on_train_epoch_end(
                train_outputs
            )

            # =================================================
            # VALIDATION
            # =================================================

            with torch.no_grad():

                self.on_validation_epoch_start()

                val_outputs = []

                for batch_id in range(
                    self.num_val_iterations_per_epoch
                ):

                    val_outputs.append(
                        self.validation_step(
                            next(self.dataloader_val)
                        )
                    )

                    step = batch_id + 1

                    if (
                        step % self.progress_every == 0
                        or
                        step == self.num_val_iterations_per_epoch
                    ):
                        pct = (
                            100.0
                            * step
                            / self.num_val_iterations_per_epoch
                        )

                        print(
                            f"[VAL] "
                            f"Epoch {self.current_epoch} | "
                            f"{step}/"
                            f"{self.num_val_iterations_per_epoch} "
                            f"({pct:.1f}%)",
                            flush=True,
                        )

                self.on_validation_epoch_end(
                    val_outputs
                )

            # =================================================
            # nnU-Net STANDARD EPOCH END
            #
            # - loss / dice logging
            # - epoch time
            # - checkpoint_latest
            # - checkpoint_best
            # - progress.png
            # - current_epoch + 1
            # =================================================

            self.on_epoch_end()

            # =================================================
            # EARLY STOPPING
            # =================================================

            if self._should_early_stop():

                self.print_to_log_file(
                    "\n=============================================="
                )
                self.print_to_log_file(
                    "EARLY STOPPING TRIGGERED"
                )
                self.print_to_log_file(
                    f"No sufficient improvement for "
                    f"{self.early_stopping_patience} epochs."
                )
                self.print_to_log_file(
                    "==============================================\n"
                )

                break

        # nnU-Net standard final checkpoint +
        # dataloader shutdown
        self.on_train_end()
