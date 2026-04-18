from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture
def tmp_path() -> Path:
    """Workspace-local tmp_path fixture to avoid host Temp ACL issues."""
    base_dir = Path("pytest_tmp_work")
    base_dir.mkdir(parents=True, exist_ok=True)
    created = base_dir / f"ed-{uuid4().hex}"
    created.mkdir(parents=True, exist_ok=False)
    try:
        yield created
    finally:
        shutil.rmtree(created, ignore_errors=True)
