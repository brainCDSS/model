
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from nnunetv2.training.loss.deep_supervision import (
    DeepSupervisionWrapper,
)

from nnUNetTrainerAneurysmEarlyStopping import (
    nnUNetTrainerAneurysmEarlyStopping,
)


# ============================================================
# GT-PRESENT-ONLY DICE + CROSS ENTROPY
#
# 기존 nnU-Net multiclass Dice:
#   foreground 52개 channel 전체를 평균
#
# 이번 수정:
#   현재 batch GT에 실제 존재하는 foreground class만
#   Dice 평균에 사용
#
# CE는 모든 voxel / 모든 53 class에 그대로 적용.
#
# GT에 foreground가 하나도 없는 batch:
#   Dice = 사용하지 않음
#   CE만 사용
#
# 목적:
#   GT-absent class가 Dice를 인위적으로 좋게 만드는 현상을 제거
# ============================================================

class GTPresentOnlyDiceCELoss(nn.Module):

    def __init__(
        self,
        ignore_label=None,
        smooth: float = 1e-5,
        weight_ce: float = 1.0,
        weight_dice: float = 1.0,
    ):

        super().__init__()

        self.ignore_label = ignore_label
        self.smooth = float(smooth)

        self.weight_ce = float(
            weight_ce
        )

        self.weight_dice = float(
            weight_dice
        )


    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        # ----------------------------------------------------
        # Target shape
        # nnU-Net:
        # B x 1 x D x H x W
        # ----------------------------------------------------

        if (
            target.ndim == logits.ndim
            and target.shape[1] == 1
        ):

            labels = target[:, 0].long()

        elif (
            target.ndim
            == logits.ndim - 1
        ):

            labels = target.long()

        else:

            raise RuntimeError(
                "Unexpected target shape: "
                f"logits={tuple(logits.shape)}, "
                f"target={tuple(target.shape)}"
            )


        num_classes = int(
            logits.shape[1]
        )

        if num_classes != 53:

            raise RuntimeError(
                "A-N00R1 Dataset501은 "
                "53 output channels가 필요합니다. "
                f"Actual={num_classes}"
            )


        # ----------------------------------------------------
        # VALID / IGNORE MASK
        # ----------------------------------------------------

        if self.ignore_label is None:

            valid = torch.ones_like(
                labels,
                dtype=torch.bool,
            )

            safe_labels = labels

        else:

            valid = (
                labels
                != int(self.ignore_label)
            )

            safe_labels = labels.clone()

            safe_labels[
                ~valid
            ] = 0


        valid_count = valid.sum()

        if valid_count == 0:

            return (
                logits.sum()
                * 0.0
            )


        # ----------------------------------------------------
        # CROSS ENTROPY
        #
        # CE는 absent class도 정상적으로 경쟁하므로
        # 기존 multiclass supervision 역할 유지.
        # ----------------------------------------------------

        if self.ignore_label is None:

            ce_loss = F.cross_entropy(
                logits,
                safe_labels,
            )

        else:

            ce_voxel = F.cross_entropy(
                logits,
                safe_labels,
                reduction="none",
            )

            ce_loss = (
                ce_voxel[
                    valid
                ].mean()
            )


        # ----------------------------------------------------
        # GT-PRESENT-ONLY SOFT DICE
        # ----------------------------------------------------

        probs = torch.softmax(
            logits,
            dim=1,
        )

        present = torch.unique(
            labels[
                valid
            ]
        )

        # background 0 제외
        present = present[
            (present > 0)
            &
            (present < num_classes)
        ]


        # ----------------------------------------------------
        # foreground가 전혀 없는 patch:
        #
        # 기존 방식처럼 absent Dice로 reward하지 않고
        # CE만 사용.
        # ----------------------------------------------------

        if present.numel() == 0:

            dice_loss = (
                logits.sum()
                * 0.0
            )

        else:

            dice_values = []

            valid_float = (
                valid.float()
            )


            for class_id in present.tolist():

                class_id = int(
                    class_id
                )

                gt_class = (
                    (labels == class_id)
                    &
                    valid
                )

                gt_float = (
                    gt_class.float()
                )

                pred_class = (
                    probs[
                        :,
                        class_id,
                    ]
                    * valid_float
                )


                intersection = (
                    pred_class
                    * gt_float
                ).sum(
                    dtype=torch.float32
                )


                pred_sum = (
                    pred_class
                ).sum(
                    dtype=torch.float32
                )


                gt_sum = (
                    gt_float
                ).sum(
                    dtype=torch.float32
                )


                dice = (
                    2.0 * intersection
                    + self.smooth
                ) / (
                    pred_sum
                    + gt_sum
                    + self.smooth
                )


                dice_values.append(
                    dice
                )


            mean_present_dice = (
                torch.stack(
                    dice_values
                ).mean()
            )


            # nnU-Net Dice loss와 동일하게
            # Dice를 음수로 사용
            dice_loss = (
                -mean_present_dice
            )


        total = (
            self.weight_ce
            * ce_loss
            +
            self.weight_dice
            * dice_loss
        )

        return total


# ============================================================
# CORRECTED TRAINER
# ============================================================

class nnUNetTrainerAneurysmSparse53Retrain(
    nnUNetTrainerAneurysmEarlyStopping
):

    """
    A-N00R1 corrected retraining.

    Network:
        unchanged PlainConvUNet

    Dataset:
        Dataset501
        background + 52 Location classes

    Changes:
        GT-present-only Dice + CE
        probabilistic foreground oversampling
        delayed EarlyStopping
    """


    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device(
            "cuda"
        ),
    ):

        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            device=device,
        )


        if self.label_manager.has_regions:

            raise RuntimeError(
                "A-N00R1은 mutually-exclusive "
                "Location classes 전용입니다."
            )


        # ====================================================
        # FOREGROUND SAMPLING
        #
        # 기존:
        #   0.33 / deterministic
        #
        # 수정:
        #   각 sample 독립적으로 75% 확률 foreground forcing
        # ====================================================

        self.oversample_foreground_percent = float(
            os.environ.get(
                "A_N00R1_OVERSAMPLE",
                "0.75",
            )
        )

        self.probabilistic_oversampling = True


        # ====================================================
        # EARLY STOPPING
        #
        # 기존 run:
        # epoch 0부터 patience 30
        # -> epoch 31 종료
        #
        # 수정:
        # 최소 epoch 80까지 학습
        # 그 이후 patience 40
        # ====================================================

        self.early_stopping_start_epoch = int(
            os.environ.get(
                "A_N00R1_EARLY_STOP_START",
                "80",
            )
        )

        self.early_stopping_patience = int(
            os.environ.get(
                "A_N00R1_PATIENCE",
                "40",
            )
        )

        self.early_stopping_min_delta = float(
            os.environ.get(
                "A_N00R1_MIN_DELTA",
                "0.0001",
            )
        )


        self.print_to_log_file(
            "\n=============================================="
        )

        self.print_to_log_file(
            "A-N00R1 SPARSE-53 CORRECTED TRAINING"
        )

        self.print_to_log_file(
            "Loss                     : "
            "GT-present-only Dice + CE"
        )

        self.print_to_log_file(
            "Foreground oversampling  : "
            f"{self.oversample_foreground_percent}"
        )

        self.print_to_log_file(
            "Probabilistic oversample : True"
        )

        self.print_to_log_file(
            "EarlyStopping start      : "
            f"{self.early_stopping_start_epoch}"
        )

        self.print_to_log_file(
            "EarlyStopping patience   : "
            f"{self.early_stopping_patience}"
        )

        self.print_to_log_file(
            "=============================================="
        )


    # ========================================================
    # LOSS
    # ========================================================

    def _build_loss(
        self,
    ):

        loss = (
            GTPresentOnlyDiceCELoss(
                ignore_label=(
                    self.label_manager
                    .ignore_label
                ),
                smooth=1e-5,
                weight_ce=1.0,
                weight_dice=1.0,
            )
        )


        # ----------------------------------------------------
        # nnU-Net 2.8.1과 동일한
        # deep-supervision weighting 유지
        # ----------------------------------------------------

        if self.enable_deep_supervision:

            deep_supervision_scales = (
                self._get_deep_supervision_scales()
            )

            weights = np.array(
                [
                    1 / (2 ** i)
                    for i
                    in range(
                        len(
                            deep_supervision_scales
                        )
                    )
                ],
                dtype=np.float64,
            )


            # single GPU + compile disabled 환경
            weights[-1] = 0.0

            weights = (
                weights
                / weights.sum()
            )


            loss = DeepSupervisionWrapper(
                loss,
                weights,
            )


        return loss
