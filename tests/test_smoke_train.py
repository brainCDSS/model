from pathlib import Path

from brainon_ai.common.config import load_config
from brainon_ai.data.synthetic import make_synthetic
from brainon_ai.engine.trainer import train


def test_cpu_synthetic_one_epoch(isolated_roots):
    manifest = make_synthetic(isolated_roots / "synthetic", "hemorrhage", cases=3, shape=(8, 8, 8))[
        0
    ]
    config = load_config(Path("configs/experiments/hemorrhage/H-S00-smoke.yaml"))
    config["data"]["manifest"] = str(manifest)
    config["training"]["epochs"] = 1
    directory = train(config)
    assert (directory / "checkpoints/best_balanced.ckpt").exists()
    assert (directory / "metrics/summary.json").exists()
