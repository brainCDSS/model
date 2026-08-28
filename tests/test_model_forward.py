import torch

from brainon_ai.models.common.tiny3d import TinySegNet


def test_model_forward_one_and_two_channels():
    assert TinySegNet(1)(torch.zeros(1, 1, 8, 8, 8)).shape == (1, 1, 8, 8, 8)
    assert TinySegNet(2)(torch.zeros(1, 2, 8, 8, 8)).shape == (1, 1, 8, 8, 8)
