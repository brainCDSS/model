from __future__ import annotations

import json
import shutil
from pathlib import Path

from brainon_ai.common.checksums import sha256_file
from brainon_ai.common.runtime import git_commit, repository_root
from brainon_ai.engine.trainer import find_run


def export_model(run_id: str, version: str) -> Path:
    run = find_run(run_id)
    task = run.parents[1].name
    destination = repository_root() / "outputs/releases" / task / version
    destination.mkdir(parents=True, exist_ok=True)
    model = destination / "model.ckpt"
    shutil.copy2(run / "checkpoints/best_balanced.ckpt", model)
    manifest = {
        "task": task,
        "model_version": version,
        "gcs_uri": f"${{BRAINON_MODEL_URI}}/releases/{task}/{version}/model.ckpt",
        "sha256": sha256_file(model),
        "input_modality": [],
        "preprocessing": {},
        "postprocessing": {},
        "labels": {"0": "background", "1": "foreground"},
        "source_repository": "https://github.com/brainCDSS/model",
        "source_git_commit": git_commit(),
        "data_release": json.loads((run / "data_release.json").read_text()),
        "metrics": json.loads((run / "metrics/summary.json").read_text()),
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return destination
