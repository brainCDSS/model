from pathlib import Path

from brainon_ai.nnunet.config import load_config


def test_h_n30_is_default_attention_ablation():
    config = load_config(
        Path(
            "configs/experiments/hemorrhage/"
            "H-N30.yaml"
        )
    )

    assert config["experiment"]["dataset_id"] == 1
    assert config["plans"]["base_identifier"] == "nnUNetPlans"
    assert (
        config["plans"]["output_identifier"]
        == "ICHPlansDefaultAttentionV1"
    )
    assert config["network"]["name"] == "default_attention"