from brainon_ai.nnunet.trainers.configurable_trainer import (
    nnUNetTrainerICHBaseline,
)


class nnUNetTrainerIschemiaComponentSampling(nnUNetTrainerICHBaseline):
    """I-N40: I-N20 residual encoder with component-centered sampling only."""
