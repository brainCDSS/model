from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class StorageBackend(ABC):
    @abstractmethod
    def list(self, uri: str) -> list[str]: ...
    @abstractmethod
    def exists(self, uri: str) -> bool: ...
    @abstractmethod
    def download(self, uri: str, destination: Path, expected_sha256: str | None = None) -> Path: ...
    @abstractmethod
    def upload(self, source: Path, uri: str, dry_run: bool = False) -> str: ...
