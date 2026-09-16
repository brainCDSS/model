from brainon_ai.nnunet.trainers.configurable_trainer import (
    nnUNetTrainerICHBaseline,
)


class nnUNetTrainerIschemiaResidualEncoder(nnUNetTrainerICHBaseline):
    """I-N20: ISLES2022 DWI+ADC nnU-Net with a residual encoder."""
