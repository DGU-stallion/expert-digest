"""Input loaders for local article sources."""

from expert_digest.ingest.loader import CrawlerLoader, list_platforms, load_crawler_documents, register_loader

# Import all loaders to trigger @register_loader decoration
from expert_digest.ingest import zhihu_loader  # noqa: F401
from expert_digest.ingest import xueqiu_loader  # noqa: F401

__all__ = [
    "CrawlerLoader",
    "list_platforms",
    "load_crawler_documents",
    "register_loader",
]
