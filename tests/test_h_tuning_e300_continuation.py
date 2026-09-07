from pathlib import Path
import yaml

from brainon_ai.nnunet.config import config_digest, load_config
from brainon_ai.nnunet.runner import prepare_continuation_checkpoint


EXPECTED = {
    "H-T01-E300": ("H-T01", 180),
    "H-T02-E300": ("H-T02", 192),
    "H-T03-E300": ("H-T03", 188),
    "H-T04-E300": ("H-T04", 204),
}


def test_e300_configs_are_fixed_length_continuations():
    for name, (parent, epoch) in EXPECTED.items():
        path = Path(f"configs/experiments/hemorrhage/tuning/{name}.yaml")
        config = load_config(path)
        assert config["trainer"]["epochs"] == 300
        assert config["trainer"]["save_every"] == 1
        assert config["pruning"]["enabled"] is False
        assert config["full_validation"]["interval"] == 5
        assert config["continuation"]["parent_experiment"] == parent
        assert config["continuation"]["expected_parent_epoch"] == epoch
        assert config["plans"]["base_identifier"] != config["plans"]["output_identifier"]


def test_prepare_continuation_checkpoint_preserves_training_state_and_resets_selection():
    config = yaml.safe_load(Path("configs/experiments/hemorrhage/tuning/H-T01-E300.yaml").read_text())
    checkpoint = {
        "current_epoch": 180,
        "optimizer_state": {"state": {1: {"momentum_buffer": [1]}}},
        "logging": {"train_losses": [1.0]},
        "_best_ema": 0.8,
        "ich_state": {
            "best_dice": 0.8,
            "best_recall": 0.9,
            "best_composite": 0.85,
            "best_full_macro_dice": 0.61,
            "best_full_macro_epoch": 176,
            "pruning": {"best": 0.85, "bad_checks": 30},
        },
    }
    optimizer_state = checkpoint["optimizer_state"]
    logging = checkpoint["logging"]
    result = prepare_continuation_checkpoint(checkpoint, config, config_digest(config))
    assert result["current_epoch"] == 180
    assert result["optimizer_state"] is optimizer_state
    assert result["logging"] is logging
    assert result["_best_ema"] is None
    assert result["ich_state"]["best_full_macro_epoch"] is None
    assert result["ich_state"]["pruning"]["bad_checks"] == 0
