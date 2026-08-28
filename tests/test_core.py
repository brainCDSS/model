from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from segresnet_baseline.checkpointing import build_checkpoint, config_sha256, load_checkpoint, save_checkpoint
from segresnet_baseline.inference import evaluate_loader
from segresnet_baseline.metrics import binary_case_metrics, summarize_cases
from segresnet_baseline.pruning import EarlyStopping


class MetricTests(unittest.TestCase):
    def test_positive_metrics_and_empty_cases_are_separate(self):
        cases = [
            ("A", [1, 0], [1, 0]),
            ("A", [1, 0], [1, 1]),
            ("B", [0], [1]),
            ("B", [0, 0], [0, 0]),
            ("B", [1, 0], [0, 0]),
        ]
        rows = []
        for index, (source, pred, target) in enumerate(cases):
            row = binary_case_metrics(torch.tensor(pred), torch.tensor(target))
            row.update({"source": source, "case_id": str(index)})
            rows.append(row)
        summary = summarize_cases(rows, dice_weight=0.5, recall_weight=0.5)
        self.assertEqual(summary["positive_cases"], 3)
        self.assertEqual(summary["empty_gt_cases"], 2)
        self.assertAlmostEqual(summary["dice_positive_macro"], 5 / 9)
        self.assertAlmostEqual(summary["recall_positive_macro"], 0.5)
        self.assertAlmostEqual(summary["empty_gt_false_positive_rate"], 0.5)
        self.assertAlmostEqual(summary["composite"], (5 / 9 + 0.5) / 2)

    def test_sliding_window_evaluation_uses_full_volume(self):
        class ForegroundModel(torch.nn.Module):
            def forward(self, image):
                return torch.cat((torch.zeros_like(image), torch.ones_like(image)), dim=1)

        image = torch.zeros(1, 1, 10, 12, 9)
        label = torch.zeros(1, 1, 10, 12, 9, dtype=torch.long)
        label[:, :, 4, 5, 3] = 1
        loader = [
            {
                "image": image,
                "label": label,
                "source": ["A"],
                "case_id": ["case"],
                "group_id": ["group"],
                "image_path": ["unused.nii.gz"],
                "label_path": ["unused.nii.gz"],
            }
        ]
        config = {
            "training": {"amp": False},
            "inference": {
                "patch_size": [8, 8, 8],
                "sliding_window_batch_size": 1,
                "overlap": 0.25,
                "blend_mode": "gaussian",
            },
            "checkpoint": {"dice_weight": 0.5, "recall_weight": 0.5},
        }
        summary, per_case = evaluate_loader(
            ForegroundModel(), loader, config, torch.device("cpu"), description="test"
        )
        self.assertEqual(len(per_case), 1)
        self.assertEqual(summary["positive_cases"], 1)
        self.assertEqual(summary["recall_positive_macro"], 1.0)


class PruningTests(unittest.TestCase):
    def test_patience_starts_after_min_epochs(self):
        pruning = EarlyStopping(enabled=True, min_epochs=3, patience=2, min_delta=0.01)
        self.assertFalse(pruning.update(0.5, 1))
        self.assertFalse(pruning.update(0.5, 2))
        self.assertEqual(pruning.bad_checks, 0)
        self.assertFalse(pruning.update(0.5, 3))
        self.assertEqual(pruning.bad_checks, 1)
        self.assertTrue(pruning.update(0.5, 4))


class CheckpointTests(unittest.TestCase):
    def test_config_hash_ignores_machine_paths_but_not_patch(self):
        first = {
            "experiment": {"device": "auto", "output_dir": "run", "manifest_path": "a.csv"},
            "data": {"root_dir": "C:/data", "patch_size": [16, 16, 8]},
            "training": {"resume": None},
            "inference": {"save_predictions": True},
        }
        moved = {
            "experiment": {"device": "cuda:1", "output_dir": "/run", "manifest_path": "/a.csv"},
            "data": {"root_dir": "/data", "patch_size": [16, 16, 8]},
            "training": {"resume": "auto"},
            "inference": {"save_predictions": False},
        }
        self.assertEqual(config_sha256(first), config_sha256(moved))
        moved["data"]["patch_size"] = [32, 16, 8]
        self.assertNotEqual(config_sha256(first), config_sha256(moved))

    def test_checkpoint_round_trip(self):
        model = torch.nn.Conv3d(1, 2, kernel_size=1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        payload = build_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=2,
            global_step=17,
            best_metrics={"dice": 0.7, "recall": 0.8, "composite": 0.75},
            pruning_state={"best": 0.75, "bad_checks": 2},
            config={"experiment": {"name": "test"}},
            manifest_hash="abc",
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            save_checkpoint(payload, path)
            restored = load_checkpoint(path, torch.device("cpu"))
        self.assertEqual(restored["epoch"], 2)
        self.assertEqual(restored["global_step"], 17)
        self.assertEqual(restored["pruning_state"]["bad_checks"], 2)
        self.assertEqual(restored["manifest_sha256"], "abc")


if __name__ == "__main__":
    unittest.main()
