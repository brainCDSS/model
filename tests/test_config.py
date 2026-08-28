from pathlib import Path

from brainon_ai.common.config import load_config


def test_config_extends_and_resolves_environment(monkeypatch):
    monkeypatch.setenv("BRAINON_CACHE_ROOT", "/tmp/brainon-test")
    config = load_config(Path("configs/experiments/hemorrhage/H-S10.yaml"))
    assert config["experiment_id"] == "H-S10"
    assert config["model"]["in_channels"] == 1
    assert config["data"]["manifest"].startswith("/tmp/brainon-test")
