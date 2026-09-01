
from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from time import time

import numpy as np
import torch
import yaml
import sys
from tqdm import tqdm

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerIschemiaBaseline(nnUNetTrainer):
    """
    I-N00 ischemia nnU-Net baseline.

    기본 nnU-Net의
    - network
    - loss
    - optimizer
    - scheduler
    - augmentation
    은 그대로 유지한다.

    추가 기능
    - YAML 기반 최대 epoch
    - Validation Dice 기반 Early Stopping
    - Best checkpoint
    - Latest checkpoint 매 epoch 저장
    - Google Drive checkpoint 백업
    - --c Resume
    - 전체 Epoch tqdm
    - Train iteration tqdm
    - Validation iteration tqdm
    - Loss / Dice / ETA 출력
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        # -------------------------------------------------
        # nnU-Net 기본 초기화
        # -------------------------------------------------
        super().__init__(
            plans,
            configuration,
            fold,
            dataset_json,
            device,
        )

        # -------------------------------------------------
        # I-N00.yaml 읽기
        # -------------------------------------------------
        repo_root = Path(__file__).resolve().parents[4]

        config_path = Path(
            os.environ.get(
                "BRAINON_EXPERIMENT_CONFIG",
                repo_root
                / "configs"
                / "experiments"
                / "ischemia"
                / "I-N00.yaml",
            )
        )

        cfg = yaml.safe_load(
            config_path.read_text(encoding="utf-8")
        )

        training_cfg = cfg["training"]
        early_cfg = training_cfg["early_stopping"]
        checkpoint_cfg = cfg["checkpoint"]

        # -------------------------------------------------
        # 학습 설정
        # -------------------------------------------------
        self.num_epochs = int(
            training_cfg["epochs"]
        )

        self.early_stopping_enabled = bool(
            early_cfg["enabled"]
        )

        self.early_stopping_patience = int(
            early_cfg["patience"]
        )

        self.early_stopping_min_delta = float(
            early_cfg["min_delta"]
        )

        self.save_latest_every = int(
            checkpoint_cfg.get(
                "save_latest_every",
                1,
            )
        )

        # -------------------------------------------------
        # Early Stopping 상태
        # -------------------------------------------------
        self.best_val_dice = float("-inf")
        self.best_val_dice_epoch = None

        self.bad_epochs = 0
        self.stop_requested = False

        # Epoch 시간 기록
        self._epoch_durations = []

        # 화면 표시용
        self._last_train_loss = float("nan")
        self._last_val_loss = float("nan")
        self._last_val_dice = float("nan")

        # -------------------------------------------------
        # Google Drive checkpoint 경로
        # -------------------------------------------------
        persistent_root = os.environ.get(
            "BRAINON_CHECKPOINT_DIR"
        )

        self.persistent_dir = (
            Path(persistent_root) / f"fold_{fold}"
            if persistent_root
            else None
        )

        if self.persistent_dir is not None:
            self.persistent_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # 이전 강제 중단 때 남은 tmp 정리
            for tmp_path in self.persistent_dir.glob("*.tmp"):
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

        self.state_filename = "early_stopping_state.json"

        # -------------------------------------------------
        # 새 Colab Runtime이라 local checkpoint가 없으면
        # Google Drive에서 복구
        # -------------------------------------------------
        self._restore_from_drive_if_needed()

        # Early Stopping 상태 복구
        self._load_early_stopping_state()

    # =====================================================
    # 시간 표시
    # =====================================================

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        if not math.isfinite(seconds):
            return "-"

        seconds = max(0, int(seconds))

        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        return (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{seconds:02d}"
        )

    # =====================================================
    # Google Drive 저장
    # =====================================================

    @staticmethod
    def _atomic_copy(
        src: Path,
        dst: Path,
    ) -> None:
        dst.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        tmp = dst.with_name(
            dst.name + ".tmp"
        )

        shutil.copy2(
            src,
            tmp,
        )

        os.replace(
            tmp,
            dst,
        )

    def _persist_file(
        self,
        filename: str,
    ) -> None:
        if self.persistent_dir is None:
            return

        src = (
            Path(self.output_folder)
            / filename
        )

        if not src.exists():
            return

        dst = (
            self.persistent_dir
            / filename
        )

        self._atomic_copy(
            src,
            dst,
        )

    def _restore_from_drive_if_needed(
        self,
    ) -> None:
        """
        Colab /content가 초기화된 경우
        Drive checkpoint를 local nnUNet_results로 복구한다.
        """
        if self.persistent_dir is None:
            return

        local_dir = Path(
            self.output_folder
        )

        local_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for filename in (
            "checkpoint_latest.pth",
            "checkpoint_best.pth",
            "checkpoint_final.pth",
            self.state_filename,
        ):
            src = (
                self.persistent_dir
                / filename
            )

            dst = (
                local_dir
                / filename
            )

            # local 파일이 없을 때만 Drive에서 복구
            if src.exists() and not dst.exists():
                shutil.copy2(
                    src,
                    dst,
                )

                print(
                    f"[Resume] Drive → Local 복구: "
                    f"{filename}"
                )

    # =====================================================
    # Early Stopping 상태 저장 / 복구
    # =====================================================

    def _save_early_stopping_state(
        self,
        completed_epoch: int,
    ) -> None:
        payload = {
            "completed_epoch": completed_epoch,
            "best_val_dice": (
                self.best_val_dice
                if math.isfinite(self.best_val_dice)
                else None
            ),
            "best_val_dice_epoch": (
                self.best_val_dice_epoch
            ),
            "bad_epochs": self.bad_epochs,
            "patience": (
                self.early_stopping_patience
            ),
            "min_delta": (
                self.early_stopping_min_delta
            ),
        }

        local_path = (
            Path(self.output_folder)
            / self.state_filename
        )

        tmp_path = local_path.with_name(
            local_path.name + ".tmp"
        )

        tmp_path.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        os.replace(
            tmp_path,
            local_path,
        )

        self._persist_file(
            self.state_filename
        )

    def _load_early_stopping_state(
        self,
    ) -> None:
        path = (
            Path(self.output_folder)
            / self.state_filename
        )

        if not path.exists():
            return

        try:
            state = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception:
            return

        best = state.get(
            "best_val_dice"
        )

        if best is not None:
            self.best_val_dice = float(best)

        self.best_val_dice_epoch = state.get(
            "best_val_dice_epoch"
        )

        self.bad_epochs = int(
            state.get(
                "bad_epochs",
                0,
            )
        )

    # =====================================================
    # nnU-Net checkpoint + Drive 백업
    # =====================================================

    def save_checkpoint(
        self,
        filename: str,
    ) -> None:
        """
        nnU-Net 기본 checkpoint에는
        network / optimizer / scaler / current_epoch /
        logger 상태가 저장된다.
        """

        super().save_checkpoint(
            filename
        )

        path = Path(filename)

        if path.exists():
            self._persist_file(
                path.name
            )

    def load_checkpoint(
        self,
        filename_or_checkpoint,
    ) -> None:
        # nnU-Net 기본 Resume
        super().load_checkpoint(
            filename_or_checkpoint
        )

        # Early Stopping 상태도 복구
        self._load_early_stopping_state()

        print(
            f"[Resume] 현재 Epoch: "
            f"{self.current_epoch}"
        )

        if math.isfinite(
            self.best_val_dice
        ):
            print(
                f"[Resume] Best Dice: "
                f"{self.best_val_dice:.4f}"
            )

        print(
            f"[Resume] Early Stopping: "
            f"{self.bad_epochs}/"
            f"{self.early_stopping_patience}"
        )

    # =====================================================
    # Epoch 종료
    # =====================================================

    def on_epoch_end(
        self,
    ):
        # -------------------------------------------------
        # nnU-Net logger에 epoch 종료 시각 저장
        # -------------------------------------------------
        self.logger.log(
            "epoch_end_timestamps",
            time(),
            self.current_epoch,
        )

        # -------------------------------------------------
        # 현재 Epoch 결과
        # -------------------------------------------------
        train_loss = float(
            np.mean(
                self.logger.get_value(
                    "train_losses",
                    step=-1,
                )
            )
        )

        val_loss = float(
            np.mean(
                self.logger.get_value(
                    "val_losses",
                    step=-1,
                )
            )
        )

        val_dice = float(
            self.logger.get_value(
                "mean_fg_dice",
                step=-1,
            )
        )

        learning_rate = float(
            self.optimizer.param_groups[0]["lr"]
        )

        start_time = float(
            self.logger.get_value(
                "epoch_start_timestamps",
                step=-1,
            )
        )

        end_time = float(
            self.logger.get_value(
                "epoch_end_timestamps",
                step=-1,
            )
        )

        epoch_seconds = (
            end_time - start_time
        )

        self._epoch_durations.append(
            epoch_seconds
        )

        self._last_train_loss = train_loss
        self._last_val_loss = val_loss
        self._last_val_dice = val_dice

        completed_epoch = (
            self.current_epoch + 1
        )

        # -------------------------------------------------
        # Best Dice 판단
        # -------------------------------------------------
        improved = (
            math.isfinite(val_dice)
            and (
                not math.isfinite(
                    self.best_val_dice
                )
                or (
                    val_dice
                    >
                    self.best_val_dice
                    +
                    self.early_stopping_min_delta
                )
            )
        )

        if improved:
            self.best_val_dice = val_dice
            self.best_val_dice_epoch = (
                completed_epoch
            )
            self.bad_epochs = 0

            # Best checkpoint
            self.save_checkpoint(
                str(
                    Path(self.output_folder)
                    / "checkpoint_best.pth"
                )
            )

        else:
            self.bad_epochs += 1

        # -------------------------------------------------
        # Latest checkpoint - 매 Epoch
        # -------------------------------------------------
        if (
            completed_epoch
            % self.save_latest_every
            == 0
        ):
            self.save_checkpoint(
                str(
                    Path(self.output_folder)
                    / "checkpoint_latest.pth"
                )
            )

        # Early stopping state 저장
        self._save_early_stopping_state(
            completed_epoch
        )

        # -------------------------------------------------
        # 전체 ETA 계산
        # -------------------------------------------------
        average_epoch_seconds = (
            sum(self._epoch_durations)
            / len(self._epoch_durations)
        )

        remaining_epochs = max(
            0,
            self.num_epochs
            - completed_epoch,
        )

        estimated_left = (
            average_epoch_seconds
            * remaining_epochs
        )

        # -------------------------------------------------
        # Epoch 결과 출력
        # -------------------------------------------------
        summary = (
            "\n"
            + "=" * 60
            + "\n"
            + f"Epoch {completed_epoch}/{self.num_epochs} 완료\n"
            + "=" * 60
            + "\n"
            + f"Train Loss        : {train_loss:.4f}\n"
            + f"Val Loss          : {val_loss:.4f}\n"
            + f"Validation Dice   : {val_dice:.4f}\n"
            + f"Best Dice         : {self.best_val_dice:.4f}"
            + f" (Epoch {self.best_val_dice_epoch})\n"
            + f"Early Stopping    : "
            + f"{self.bad_epochs}/"
            + f"{self.early_stopping_patience}\n"
            + f"Learning Rate     : {learning_rate:.8f}\n"
            + f"Epoch Time        : "
            + f"{self._format_seconds(epoch_seconds)}\n"
            + f"Estimated Left    : "
            + f"{self._format_seconds(estimated_left)}\n"
            + "=" * 60
        )

        # training_log 파일에 기록
        self.print_to_log_file(
            summary,
            also_print_to_console=False,
            add_timestamp=False,
        )

        # tqdm 상태바 안 깨지게 출력
        tqdm.write(
            summary
        )

        if improved:
            tqdm.write(
                "★ New Best Validation Dice "
                "→ checkpoint_best.pth 저장 완료"
            )

        tqdm.write(
            "✓ checkpoint_latest.pth "
            "→ Google Drive 백업 완료"
        )

        # nnU-Net progress.png
        if self.local_rank == 0:
            self.logger.plot_progress_png(
                self.output_folder
            )

        # -------------------------------------------------
        # Early Stopping 판단
        # -------------------------------------------------
        if (
            self.early_stopping_enabled
            and self.bad_epochs
            >= self.early_stopping_patience
        ):
            self.stop_requested = True

            message = (
                "\nEarly Stopping triggered.\n"
                f"Best Dice  : "
                f"{self.best_val_dice:.4f}\n"
                f"Best Epoch : "
                f"{self.best_val_dice_epoch}\n"
            )

            self.print_to_log_file(
                message,
                also_print_to_console=False,
                add_timestamp=False,
            )

            tqdm.write(
                message
            )

        # nnU-Net epoch index 증가
        self.current_epoch += 1

    # =====================================================
    # 실제 학습 Loop
    # =====================================================

    def _progress_bar(
        self,
        current,
        total,
        width=30,
    ):
        """Colab subprocess에서도 확실히 보이는 텍스트 게이지."""
        ratio = current / total if total else 0
        filled = int(width * ratio)

        bar = (
            "█" * filled
            + "░" * (width - filled)
        )

        percent = ratio * 100

        return (
            f"[{bar}] "
            f"{current}/{total} "
            f"({percent:5.1f}%)"
        )


    def run_training(
        self,
    ):
        self.on_train_start()

        start_epoch = self.current_epoch

        print("", flush=True)
        print("=" * 70, flush=True)
        print("I-N00 nnU-Net Training", flush=True)
        print(
            f"Resume Epoch : {start_epoch + 1}",
            flush=True,
        )
        print(
            f"Maximum      : {self.num_epochs}",
            flush=True,
        )
        print(
            f"Early Stop   : "
            f"{self.early_stopping_patience}",
            flush=True,
        )
        print("=" * 70, flush=True)

        training_start_time = time()

        for epoch in range(
            start_epoch,
            self.num_epochs,
        ):
            display_epoch = epoch + 1

            # =================================================
            # 전체 학습 진행률
            # =================================================
            completed_before = epoch

            elapsed = (
                time()
                - training_start_time
            )

            if completed_before > start_epoch:
                completed_this_run = (
                    completed_before
                    - start_epoch
                )

                avg_epoch_time = (
                    elapsed
                    / completed_this_run
                )

                remaining = (
                    self.num_epochs
                    - completed_before
                )

                overall_eta = (
                    avg_epoch_time
                    * remaining
                )

                eta_text = (
                    self._format_seconds(
                        overall_eta
                    )
                )
            else:
                eta_text = "계산 중"

            print("", flush=True)
            print("=" * 70, flush=True)

            print(
                "전체 학습 "
                + self._progress_bar(
                    completed_before,
                    self.num_epochs,
                )
                + f" | ETA {eta_text}",
                flush=True,
            )

            print(
                f"Epoch {display_epoch}/"
                f"{self.num_epochs}",
                flush=True,
            )

            print("=" * 70, flush=True)

            epoch_start = time()

            # =================================================
            # Epoch 시작
            # =================================================
            self.on_epoch_start()
            self.on_train_epoch_start()

            # =================================================
            # TRAIN
            # =================================================
            train_outputs = []

            running_train_loss = 0.0

            train_start = time()

            print(
                "\n[TRAIN]",
                flush=True,
            )

            for batch_id in range(
                self.num_iterations_per_epoch
            ):
                output = self.train_step(
                    next(
                        self.dataloader_train
                    )
                )

                train_outputs.append(
                    output
                )

                loss_value = float(
                    np.mean(
                        output["loss"]
                    )
                )

                running_train_loss += (
                    loss_value
                )

                current = batch_id + 1

                avg_loss = (
                    running_train_loss
                    / current
                )

                # 10 iteration마다 출력
                # + 첫 iteration
                # + 마지막 iteration
                if (
                    current == 1
                    or current % 10 == 0
                    or current
                    == self.num_iterations_per_epoch
                ):
                    elapsed_train = (
                        time()
                        - train_start
                    )

                    sec_per_it = (
                        elapsed_train
                        / current
                    )

                    remaining_it = (
                        self.num_iterations_per_epoch
                        - current
                    )

                    eta = (
                        sec_per_it
                        * remaining_it
                    )

                    print(
                        self._progress_bar(
                            current,
                            self.num_iterations_per_epoch,
                        )
                        + f" | loss {loss_value:.4f}"
                        + f" | avg {avg_loss:.4f}"
                        + " | ETA "
                        + self._format_seconds(
                            eta
                        ),
                        flush=True,
                    )

            self.on_train_epoch_end(
                train_outputs
            )

            # =================================================
            # VALIDATION
            # =================================================
            val_outputs = []

            running_val_loss = 0.0

            val_start = time()

            print(
                "\n[VALIDATION]",
                flush=True,
            )

            with torch.no_grad():
                self.on_validation_epoch_start()

                for batch_id in range(
                    self.num_val_iterations_per_epoch
                ):
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

                    loss_value = float(
                        np.mean(
                            output["loss"]
                        )
                    )

                    running_val_loss += (
                        loss_value
                    )

                    current = batch_id + 1

                    avg_loss = (
                        running_val_loss
                        / current
                    )

                    # Validation은 5 iteration마다
                    if (
                        current == 1
                        or current % 5 == 0
                        or current
                        == self.num_val_iterations_per_epoch
                    ):
                        elapsed_val = (
                            time()
                            - val_start
                        )

                        sec_per_it = (
                            elapsed_val
                            / current
                        )

                        remaining_it = (
                            self.num_val_iterations_per_epoch
                            - current
                        )

                        eta = (
                            sec_per_it
                            * remaining_it
                        )

                        print(
                            self._progress_bar(
                                current,
                                self.num_val_iterations_per_epoch,
                            )
                            + f" | loss {loss_value:.4f}"
                            + f" | avg {avg_loss:.4f}"
                            + " | ETA "
                            + self._format_seconds(
                                eta
                            ),
                            flush=True,
                        )

                self.on_validation_epoch_end(
                    val_outputs
                )

            # =================================================
            # EPOCH 종료
            # =================================================
            self.on_epoch_end()

            print(
                "\n전체 학습 "
                + self._progress_bar(
                    self.current_epoch,
                    self.num_epochs,
                ),
                flush=True,
            )

            # Early Stopping
            if self.stop_requested:
                break

        self.on_train_end()
