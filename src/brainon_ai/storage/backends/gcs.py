from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import urlparse

from brainon_ai.common.checksums import verify_sha256
from brainon_ai.storage.backends.base import StorageBackend


class GCSBackend(StorageBackend):
    def __init__(self, retries: int = 3) -> None:
        try:
            from google.cloud import storage
        except ImportError as error:
            raise RuntimeError("Install the storage extra: pip install -e '.[storage]'") from error
        self.client = storage.Client()
        self.retries = retries

    @staticmethod
    def _parts(uri: str) -> tuple[str, str]:
        parsed = urlparse(uri)
        if parsed.scheme != "gs":
            raise ValueError("GCS URI must start with gs://")
        return parsed.netloc, parsed.path.lstrip("/")

    def list(self, uri: str) -> list[str]:
        bucket, prefix = self._parts(uri)
        return [
            f"gs://{bucket}/{blob.name}" for blob in self.client.list_blobs(bucket, prefix=prefix)
        ]

    def exists(self, uri: str) -> bool:
        bucket, name = self._parts(uri)
        return self.client.bucket(bucket).blob(name).exists()

    def _retry(self, operation):
        for attempt in range(self.retries):
            try:
                return operation()
            except Exception:
                if attempt + 1 == self.retries:
                    raise
                time.sleep(2**attempt)

    def download(self, uri: str, destination: Path, expected_sha256: str | None = None) -> Path:
        bucket, name = self._parts(uri)
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".partial")
        self._retry(lambda: self.client.bucket(bucket).blob(name).download_to_filename(partial))
        if expected_sha256 and not verify_sha256(partial, expected_sha256):
            partial.unlink()
            raise ValueError("SHA-256 mismatch")
        partial.replace(destination)
        return destination

    def upload(self, source: Path, uri: str, dry_run: bool = False) -> str:
        if dry_run:
            return uri
        bucket, name = self._parts(uri)
        self._retry(lambda: self.client.bucket(bucket).blob(name).upload_from_filename(source))
        return uri
