import numpy as np

from brainon_ai.metrics.segmentation import (
    case_metrics,
    components,
    dice_fg,
    hd95_mm,
    lesion_metrics,
)


def test_binary_dice_excludes_background():
    target = np.zeros((5, 5, 5), bool)
    target[1:3, 1:3, 1:3] = True
    assert dice_fg(target, target) == 1.0


def test_26_connectivity_and_matching():
    mask = np.zeros((3, 3, 3), bool)
    mask[0, 0, 0] = True
    mask[1, 1, 1] = True
    assert components(mask)[1] == 1
    metrics = lesion_metrics(mask, mask)
    assert metrics["lesion_f1"] == 1.0


def test_hd95_uses_spacing_mm():
    target = np.zeros((7, 7, 7), bool)
    prediction = target.copy()
    target[2, 2, 2] = True
    prediction[3, 2, 2] = True
    assert hd95_mm(prediction, target, (2.0, 1.0, 1.0)) == 2.0


def test_negative_case_is_separate():
    empty = np.zeros((4, 4, 4), bool)
    result = case_metrics(empty, empty)
    assert np.isnan(result["dice_fg_case_mean"])
    assert result["patient_specificity"] == 1.0
