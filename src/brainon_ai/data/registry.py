from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from brainon_ai.common.checksums import verify_sha256
from brainon_ai.common.runtime import cache_root, repository_root


def dataset_entries() -> list[dict[str, str]]:
    entries = []
    for path in sorted((repository_root() / "registry/datasets").glob("*.yaml")):
        import yaml

        entries.append(yaml.safe_load(path.read_text(encoding="utf-8")))
    return entries


def verify_release(release: str) -> tuple[bool, list[str]]:
    manifest = cache_root() / "synthetic" / release.replace("_v1.0.0", "") / "manifest.csv"
    matches = list((cache_root() / "synthetic").glob("*/manifest.csv"))
    if not manifest.exists():
        manifest = next(
            (p for p in matches if release in (p.parent / "READY.json").read_text()), manifest
        )
    if not manifest.exists():
        return False, [f"manifest not found for {release}"]
    failures = []
    with manifest.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not verify_sha256(Path(row["label_path"]), row["sha256"]):
                failures.append(row["case_id"])
    return not failures, failures


def pull_local_release(source: Path, release: str) -> Path:
    destination = cache_root() / "releases" / release
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    shutil.copytree(source, partial)
    partial.rename(destination)
    (destination / ".ready").write_text(json.dumps({"release": release}), encoding="utf-8")
    return destination
