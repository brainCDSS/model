#!/usr/bin/env python3
"""Generate SHA-256 and environment metadata for a preprocessing release."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--metadata-root", required=True, type=Path)
    args = parser.parse_args()
    root, metadata = args.release_root.resolve(), args.metadata_root.resolve()
    metadata.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in root.rglob("*") if p.is_file() and metadata not in p.parents)
    rows = [{"path": p.relative_to(root).as_posix(), "size_bytes": p.stat().st_size,
             "sha256": sha256(p)} for p in files]
    with (metadata / "SHA256SUMS.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "size_bytes", "sha256"])
        writer.writeheader(); writer.writerows(rows)
    versions = {}
    for name in ("numpy", "scipy", "pandas", "nibabel", "monai", "torch", "nnunetv2"):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    try: commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError): commit = None
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "git_commit": commit, "packages": versions}
    (metadata / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    summary = {"files": len(rows), "bytes": sum(r["size_bytes"] for r in rows)}
    (metadata / "checksum_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
