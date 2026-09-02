from __future__ import annotations

import csv
import json
import os
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

        self.early_stopping_monitor = (
            "val_patient_mean_dice"
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

        # Best checkpoint is intentionally independent from
        # nnU-Net's default _best_ema pseudo-Dice criterion.
        self._best_patient_mean_dice = None

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

    def _restore_patient_best_from_history(
        self,
    ) -> None:

        path = (
            self.training_history_file
        )

        if not path.exists():
            self._best_patient_mean_dice = None
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
                    "val_patient_mean_dice",
                    "",
                )

                if raw in {
                    "",
                    None,
                }:
                    continue

                try:
                    score = float(
                        raw
                    )

                except (
                    TypeError,
                    ValueError,
                ):
                    continue

                if np.isfinite(
                    score
                ):
                    scores.append(
                        score
                    )

        self._best_patient_mean_dice = (
            max(scores)
            if scores
            else None
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
                    "val_patient_mean_dice",
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

        Best checkpoint itself always represents the true maximum
        val_patient_mean_dice observed so far.
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

        patient_rows = []

        negative_cases = 0
        negative_case_fp = 0

        self.set_deep_supervision_enabled(
            False
        )

        self.network.eval()

        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,

            # Safer on a T4 with 53-class Location output:
            # network inference remains on GPU while accumulation
            # can fall back to CPU memory.
            perform_everything_on_device=False,

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

                for patient_index, case_id in enumerate(
                    dataset_val.identifiers,
                    start=1,
                ):

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

                    prediction_logits = (
                        prediction_logits.cpu()
                    )

                    output_filename_truncated = str(
                        tmp_dir
                        / case_id
                    )

                    # Same original-geometry restoration pipeline
                    # used by official nnU-Net validation.
                    export_prediction_from_logits(
                        prediction_logits,
                        properties,
                        self.configuration_manager,
                        self.plans_manager,
                        self.dataset_json,
                        output_filename_truncated,
                        False,
                    )

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
                            self.current_epoch,

                        "epoch_1based":
                            self.current_epoch + 1,

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

                    elapsed = (
                        time.perf_counter()
                        - epoch_start
                    )

                    avg_patient_seconds = (
                        elapsed
                        / patient_index
                    )

                    remaining = (
                        len(val_keys)
                        - patient_index
                    )

                    eta_seconds = (
                        avg_patient_seconds
                        * remaining
                    )

                    if (
                        patient_index
                        % self.patient_progress_every
                        == 0

                        or patient_index
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
                            * patient_index
                            / len(val_keys)
                        )

                        print(
                            "[PATIENT-VAL] "
                            f"Epoch "
                            f"{self.current_epoch + 1}/"
                            f"{self.num_epochs} | "
                            f"{patient_index}/"
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
            "\n[PATIENT-VAL SUMMARY] "
            f"Epoch {self.current_epoch + 1} | "
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

        train_loss = (
            self.logger.get_value(
                "train_losses",
                step=-1,
            )
        )

        val_loss = (
            self.logger.get_value(
                "val_losses",
                step=-1,
            )
        )

        online_dice = (
            self.logger.get_value(
                "mean_fg_dice",
                step=-1,
            )
        )

        ema_online_dice = (
            self.logger.get_value(
                "ema_fg_dice",
                step=-1,
            )
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
            "online_patch_pseudo_dice",
            np.round(
                online_dice,
                decimals=4,
            ),
            "(diagnostic only)",
        )

        if self._latest_patient_metrics is None:
            raise RuntimeError(
                "Patient full-volume validation metrics "
                "were not generated before on_epoch_end."
            )

        patient_score = float(
            self._latest_patient_metrics[
                "val_patient_mean_dice"
            ]
        )

        if not np.isfinite(
            patient_score
        ):
            raise RuntimeError(
                "val_patient_mean_dice is not finite. "
                "Best checkpoint cannot be selected."
            )

        previous_best = (
            self._best_patient_mean_dice
        )

        is_new_best = (
            previous_best is None
            or patient_score
            > previous_best
        )

        if is_new_best:

            self._best_patient_mean_dice = (
                patient_score
            )

            self.print_to_log_file(
                "[BEST] New best "
                "val_patient_mean_dice="
                f"{patient_score:.6f}"
            )

            self.save_checkpoint(
                str(
                    Path(
                        self.output_folder
                    )
                    / "checkpoint_best.pth"
                )
            )

        # Periodic/latest checkpoint
        if (
            (
                self.current_epoch + 1
            )
            % self.save_every
            == 0

            and self.current_epoch
            != (
                self.num_epochs - 1
            )
        ):

            self.save_checkpoint(
                str(
                    Path(
                        self.output_folder
                    )
                    / "checkpoint_latest.pth"
                )
            )

        epoch_seconds = (
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

        # -----------------------------------------------------
        # Persist patient/class metrics + one-row epoch history
        # -----------------------------------------------------

        self._write_epoch_outputs(
            metrics=(
                self._latest_patient_metrics
            ),
            patient_rows=(
                self._latest_patient_rows
            ),
            class_rows=(
                self._latest_class_rows
            ),
            is_best=is_new_best,
        )

        history_row = {
            "epoch_0based":
                self.current_epoch,

            "epoch_1based":
                self.current_epoch + 1,

            "monitor":
                self.early_stopping_monitor,

            "train_loss":
                float(
                    train_loss
                ),

            "online_val_loss":
                float(
                    val_loss
                ),

            "online_patch_mean_fg_dice":
                float(
                    online_dice
                ),

            "online_patch_ema_fg_dice":
                float(
                    ema_online_dice
                ),

            "val_patient_mean_dice":
                self._latest_patient_metrics[
                    "val_patient_mean_dice"
                ],

            "val_patient_median_dice":
                self._latest_patient_metrics[
                    "val_patient_median_dice"
                ],

            "val_patient_mean_macro_dice":
                self._latest_patient_metrics[
                    "val_patient_mean_macro_dice"
                ],

            "val_patient_mean_precision":
                self._latest_patient_metrics[
                    "val_patient_mean_precision"
                ],

            "val_patient_mean_recall":
                self._latest_patient_metrics[
                    "val_patient_mean_recall"
                ],

            "val_macro_dice":
                self._latest_patient_metrics[
                    "val_macro_dice"
                ],

            "val_macro_precision":
                self._latest_patient_metrics[
                    "val_macro_precision"
                ],

            "val_macro_recall":
                self._latest_patient_metrics[
                    "val_macro_recall"
                ],

            "negative_case_fp_rate":
                self._latest_patient_metrics[
                    "negative_case_fp_rate"
                ],

            "best_patient_mean_dice":
                self._best_patient_mean_dice,

            "is_best":
                int(
                    is_new_best
                ),

            "epoch_seconds":
                float(
                    epoch_seconds
                ),
        }

        self._upsert_training_history(
            history_row
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

        self.on_train_start()

        # Restore best score from CSV when resuming with --c.
        self._restore_patient_best_from_history()

        self.print_to_log_file(
            "\n=============================================="
        )

        self.print_to_log_file(
            "A-N00 Standard nnU-Net "
            "Patient-Level Validation Policy"
        )

        self.print_to_log_file(
            f"Max epochs          : "
            f"{self.num_epochs}"
        )

        self.print_to_log_file(
            f"Train iterations     : "
            f"{self.num_iterations_per_epoch}"
        )

        self.print_to_log_file(
            f"Online val iterations: "
            f"{self.num_val_iterations_per_epoch} "
            "(diagnostic only)"
        )

        self.print_to_log_file(
            f"Full-volume val cases: "
            f"{self.expected_validation_cases}"
        )

        self.print_to_log_file(
            f"Best/EarlyStop metric: "
            f"{self.early_stopping_monitor}"
        )

        self.print_to_log_file(
            f"Patience            : "
            f"{self.early_stopping_patience}"
        )

        self.print_to_log_file(
            f"Min delta           : "
            f"{self.early_stopping_min_delta}"
        )

        self.print_to_log_file(
            "==============================================\n"
        )

        for _ in range(
            self.current_epoch,
            self.num_epochs,
        ):

            self.on_epoch_start()

            # =================================================
            # TRAIN
            # =================================================

            self.on_train_epoch_start()

            train_outputs = []

            train_start = (
                time.perf_counter()
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

                step = (
                    batch_id + 1
                )

                if (
                    step
                    % self.progress_every
                    == 0

                    or step
                    == self.num_iterations_per_epoch
                ):

                    elapsed = (
                        time.perf_counter()
                        - train_start
                    )

                    eta = (
                        (
                            elapsed
                            / step
                        )
                        * (
                            self.num_iterations_per_epoch
                            - step
                        )
                    )

                    running_loss = float(
                        np.mean(
                            [
                                float(
                                    x["loss"]
                                )
                                for x
                                in train_outputs
                            ]
                        )
                    )

                    pct = (
                        100.0
                        * step
                        / self.num_iterations_per_epoch
                    )

                    print(
                        "[TRAIN] "
                        f"Epoch "
                        f"{self.current_epoch + 1}/"
                        f"{self.num_epochs} | "
                        f"{step}/"
                        f"{self.num_iterations_per_epoch} "
                        f"({pct:.1f}%) | "
                        f"loss={running_loss:.6f} | "
                        f"ETA={eta / 60.0:.1f} min",
                        flush=True,
                    )

            self.on_train_epoch_end(
                train_outputs
            )

            # =================================================
            # ONLINE PATCH VALIDATION
            #
            # Diagnostic only.
            # Does NOT select checkpoint_best.
            # =================================================

            with torch.no_grad():

                self.on_validation_epoch_start()

                val_outputs = []

                val_start = (
                    time.perf_counter()
                )

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

                    step = (
                        batch_id + 1
                    )

                    if (
                        step
                        % self.progress_every
                        == 0

                        or step
                        == self.num_val_iterations_per_epoch
                    ):

                        elapsed = (
                            time.perf_counter()
                            - val_start
                        )

                        eta = (
                            (
                                elapsed
                                / step
                            )
                            * (
                                self.num_val_iterations_per_epoch
                                - step
                            )
                        )

                        running_val_loss = float(
                            np.mean(
                                [
                                    float(
                                        x[
                                            "loss"
                                        ]
                                    )
                                    for x
                                    in val_outputs
                                ]
                            )
                        )

                        pct = (
                            100.0
                            * step
                            / self.num_val_iterations_per_epoch
                        )

                        print(
                            "[ONLINE-VAL] "
                            f"Epoch "
                            f"{self.current_epoch + 1}/"
                            f"{self.num_epochs} | "
                            f"{step}/"
                            f"{self.num_val_iterations_per_epoch} "
                            f"({pct:.1f}%) | "
                            f"loss="
                            f"{running_val_loss:.6f} | "
                            f"ETA={eta / 60.0:.1f} min",
                            flush=True,
                        )

                self.on_validation_epoch_end(
                    val_outputs
                )

            # =================================================
            # FORMAL PATIENT FULL-VOLUME VALIDATION
            # =================================================

            self._run_patient_full_volume_validation()

            # =================================================
            # EPOCH END
            #
            # checkpoint_best is selected here by
            # val_patient_mean_dice.
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
                    f"monitor="
                    f"{self.early_stopping_monitor}"
                )

                self.print_to_log_file(
                    f"No sufficient improvement "
                    f"(min_delta="
                    f"{self.early_stopping_min_delta}) "
                    f"for "
                    f"{self.early_stopping_patience} "
                    f"epochs."
                )

                self.print_to_log_file(
                    "==============================================\n"
                )

                break

        self.on_train_end()
