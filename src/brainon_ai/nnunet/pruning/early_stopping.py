from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class EarlyStopping:
    enabled: bool
    min_epochs: int
    patience: int
    min_delta: float
    best: float = float("-inf")
    bad_checks: int = 0

    def update(self, metric: float, completed_epochs: int) -> bool:
        if not math.isfinite(metric):
            raise RuntimeError(f"pruning metric이 유한한 값이 아닙니다: {metric}")
        if metric > self.best + self.min_delta:
            self.best = float(metric)
            self.bad_checks = 0
        elif completed_epochs >= self.min_epochs:
            self.bad_checks += 1
        return bool(self.enabled and completed_epochs >= self.min_epochs and self.bad_checks >= self.patience)

    def state_dict(self) -> dict:
        return {"best": float(self.best), "bad_checks": int(self.bad_checks)}

    def load_state_dict(self, state: dict | None) -> None:
        if not state:
            return
        self.best = float(state.get("best", self.best))
        self.bad_checks = int(state.get("bad_checks", self.bad_checks))
