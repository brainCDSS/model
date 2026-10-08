from brainon_ai.nnunet.trainers.configurable_trainer import (
    nnUNetTrainerICHBaseline,
)


class nnUNetTrainerIschemiaInstanceCenter(nnUNetTrainerICHBaseline):
    """I-N30: I-N20 residual encoder with instance-center loss only."""
