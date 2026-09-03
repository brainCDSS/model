from pathlib import Path

from brainon_ai.nnunet.trainers.configurable_trainer import nnUNetTrainerICHBaseline


def test_top_three_macro_epochs_and_checkpoint_pruning(tmp_path: Path):
    trainer = nnUNetTrainerICHBaseline.__new__(nnUNetTrainerICHBaseline)
    trainer._full_validation_top_k = 3
    trainer._full_validation_neighborhood_radius = 4
    trainer._archive_every_epoch = True
    trainer.local_rank = 0
    trainer.output_folder = str(tmp_path)
    trainer._full_validation_history_path = tmp_path / "full_validation_history.json"

    folder = tmp_path / "epoch_checkpoints"
    folder.mkdir()
    for epoch in range(1, 31):
        (folder / f"checkpoint_epoch_{epoch:04d}.pth").touch()

    history = [
        {"epoch": 5, "macro_dice_positive": 0.51},
        {"epoch": 10, "macro_dice_positive": 0.62},
        {"epoch": 15, "macro_dice_positive": 0.58},
        {"epoch": 20, "macro_dice_positive": 0.70},
        {"epoch": 25, "macro_dice_positive": 0.66},
        {"epoch": 30, "macro_dice_positive": 0.40},
    ]

    assert trainer._top_full_macro_epochs(history) == [20, 25, 10]
    trainer._prune_epoch_checkpoints(history, completed_epochs=30)

    retained = {int(path.stem.rsplit("_", 1)[1]) for path in folder.glob("*.pth")}
    expected = set(range(6, 15)) | set(range(16, 30)) | set(range(27, 31))
    assert retained == expected
    assert trainer._retained_checkpoint_count() == len(expected)
