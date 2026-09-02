import numpy as np

from brainon_ai.nnunet.sampling.small_lesion import build_sampling_probabilities, case_weight

CONFIG = {"volume_thresholds_ml": [1.0, 5.0, 20.0],
          "weights": {"negative": 1, "lt_1ml": 8, "1_to_5ml": 4,
                      "5_to_20ml": 2, "ge_20ml": 1}}


class Dataset:
    identifiers = ["negative", "tiny", "large"]
    def load_case(self, identifier):
        counts = {"negative": 0, "tiny": 1, "large": 30}
        seg = np.zeros((1, 1, 30), dtype=np.uint8)
        seg.flat[:counts[identifier]] = 1
        return None, seg, None, {}


def test_case_weight_groups():
    assert case_weight(0, False, CONFIG) == ("negative", 1.0)
    assert case_weight(0.5, True, CONFIG) == ("lt_1ml", 8.0)
    assert case_weight(3, True, CONFIG) == ("1_to_5ml", 4.0)
    assert case_weight(10, True, CONFIG) == ("5_to_20ml", 2.0)
    assert case_weight(25, True, CONFIG) == ("ge_20ml", 1.0)


def test_probabilities_follow_dataset_order():
    probabilities, report = build_sampling_probabilities(Dataset(), [100, 100, 1], CONFIG)
    np.testing.assert_allclose(probabilities.sum(), 1.0)
    assert probabilities[1] > probabilities[2]
    assert report["num_train_cases"] == 3
