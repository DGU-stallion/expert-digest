"""Unified crawler loader with platform dispatch."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from expert_digest.domain.models import Document

_LOADERS: dict[str, type["CrawlerLoader"]] = {}


def register_loader(loader_cls: type["CrawlerLoader"]) -> type["CrawlerLoader"]:
    """Register a CrawlerLoader subclass so it can be discovered by import-crawler."""
    _LOADERS[loader_cls.platform] = loader_cls
    return loader_cls


def load_crawler_documents(platform: str, path: str | Path) -> list[Document]:
    """Dispatch to the registered loader for *platform* and load documents."""
    if platform not in _LOADERS:
        raise ValueError(
            f"Unknown platform: {platform!r}. "
            f"Available: {', '.join(sorted(_LOADERS))}"
        )
    return _LOADERS[platform]().load(path)


def list_platforms() -> list[str]:
    """Return the list of registered platform identifiers."""
    return list(_LOADERS)


class CrawlerLoader(ABC):
    """Abstract base for platform-specific crawler loaders.

    Subclasses must set *platform* and implement :meth:`load`.
    Registration is automatic via :func:`register_loader`.
    """

    platform: str = ""

    @abstractmethod
    def load(self, path: str | Path) -> list[Document]:
        """Load documents from a crawler output directory or file."""
