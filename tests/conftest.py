from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def isolated_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("BRAINON_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("BRAINON_RUN_ROOT", str(tmp_path / "runs"))
    return tmp_path
