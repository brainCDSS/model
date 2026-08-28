from __future__ import annotations

import shutil
from pathlib import Path
from urllib.parse import urlparse

from brainon_ai.common.checksums import verify_sha256
from brainon_ai.storage.backends.base import StorageBackend


def file_path(uri: str) -> Path:
    parsed = urlparse(uri)
    return Path(parsed.path if parsed.scheme == "file" else uri).resolve()


class LocalBackend(StorageBackend):
    def list(self, uri: str) -> list[str]:
        root = file_path(uri)
        return [path.as_uri() for path in sorted(root.rglob("*")) if path.is_file()]

    def exists(self, uri: str) -> bool:
        return file_path(uri).exists()

    def download(self, uri: str, destination: Path, expected_sha256: str | None = None) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".partial")
        shutil.copy2(file_path(uri), partial)
        if expected_sha256 and not verify_sha256(partial, expected_sha256):
            partial.unlink()
            raise ValueError("SHA-256 mismatch")
        partial.replace(destination)
        return destination

    def upload(self, source: Path, uri: str, dry_run: bool = False) -> str:
        destination = file_path(uri)
        if not dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        return destination.as_uri()

    def sync(self, source: Path, uri: str, dry_run: bool = False) -> list[str]:
        uploaded = []
        for path in source.rglob("*"):
            if path.is_file():
                uploaded.append(
                    self.upload(path, str(file_path(uri) / path.relative_to(source)), dry_run)
                )
        return uploaded
