"""Default path configuration for the local MVP workspace."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem locations used by the local project."""

    raw_data: Path = Path("data/raw")
    processed_data: Path = Path("data/processed")
    vector_store: Path = Path("data/vector_store")
    outputs: Path = Path("data/outputs")


DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
DEFAULT_PATHS = ProjectPaths()
