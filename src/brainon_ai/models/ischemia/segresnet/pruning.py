"""Early stopping state with the validated I-S00 comparisons."""

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float
    best_dice: float = -1.0
    bad_epochs: int = 0

    def update(self, val_dice: float) -> bool:
        improved = val_dice > (self.best_dice + self.min_delta)
        if improved:
            self.best_dice = val_dice
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience
