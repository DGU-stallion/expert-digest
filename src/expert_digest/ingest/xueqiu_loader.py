"""Load source articles from Xueqiu (雪球) crawler export files.

TODO: Implement XueqiuLoader when the Xueqiu crawler is ready.
"""

from __future__ import annotations

from pathlib import Path

from expert_digest.domain.models import Document
from expert_digest.ingest.loader import CrawlerLoader, register_loader


@register_loader
class XueqiuLoader(CrawlerLoader):
    """CrawlerLoader placeholder for Xueqiu (雪球) platform."""

    platform = "xueqiu"

    def load(self, path: str | Path) -> list[Document]:
        raise NotImplementedError("XueqiuLoader is not yet implemented")
