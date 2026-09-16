from pathlib import Path

import pytest

from brainon_ai.common.config import load_config
from brainon_ai.nnunet.config import load_config as load_nnunet_config


@pytest.mark.parametrize(
    ("config_name", "network_name", "plans_identifier"),
    [
        ("H-N00.yaml", "default", "ICHPlansBaselineV1"),
        ("H-N10.yaml", "resenc_m", "ICHPlansResEncMV1"),
        (
            "H-N20.yaml",
            "resenc_m_attention",
            "ICHPlansResEncMAttentionV1",
        ),
    ],
)
def test_load_hemorrhage_nnunet_configs(
    config_name,
    network_name,
    plans_identifier,
):
    path = Path("configs/experiments/hemorrhage") / config_name
    config = load_nnunet_config(path)

    assert config["experiment"]["dataset_id"] == 1
    assert config["experiment"]["configuration"] == "3d_fullres"
    assert config["experiment"]["fold"] == 0
    assert config["network"]["name"] == network_name
    assert config["plans"]["output_identifier"] == plans_identifier
    assert config["plans"]["patch_size"] == [20, 320, 256]
    assert config["plans"]["batch_size"] > 0
    assert config["trainer"]["epochs"] > 0


def test_load_a_b05_component_centered_config():
    path = Path("configs/experiments/aneurysm_binary/A-B05.yaml")
    config = load_nnunet_config(path)

    assert config["experiment"]["name"] == "A-B05"
    assert config["plans"]["output_identifier"] == (
        "AneurysmBinaryPlansTverskyFocalComponentCenteredV1"
    )
    assert config["trainer"]["oversample_foreground_percent"] == 0.33
    assert config["loss"]["name"] == "tversky_focal"
    assert config["sampling"]["strategy"] == "component_center"
    assert set(config["sampling"]["component_weights"].values()) == {1.0}


@pytest.mark.parametrize(
    ("config_name", "model_name"),
    [
        ("H-S00.yaml", "segresnet"),
        ("H-S10.yaml", "segresnet_attention_gate"),
        ("H-S20.yaml", "segresnet_multiscale_context"),
    ],
)
def test_load_hemorrhage_segresnet_configs(config_name, model_name):
    path = Path("configs/experiments/hemorrhage") / config_name
    config = load_config(path)

    assert config["model"].get("name", "segresnet") == model_name
    assert config["model"]["in_channels"] == 1
    assert config["data"]["root_dir"]