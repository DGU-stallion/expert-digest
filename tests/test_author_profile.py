from pathlib import Path

import pytest

from expert_digest.domain.models import Document
from expert_digest.knowledge.author_profile import (
    build_author_profile,
    extract_author_profile_from_documents,
)


def _sample_documents() -> list[Document]:
    return [
        Document.create(
            author="黄彦臻",
            title="泡泡玛特的供给与需求复盘",
            content=(
                "首先定义问题，然后给出假设。因为供给变化，所以利润率会波动。"
                "如果估值过高，那么风险回撤会扩大。"
            ),
            source="sample",
        ),
        Document.create(
            author="黄彦臻",
            title="投资框架：风险控制与现金流",
            content=(
                "一方面关注现金流质量，另一方面关注竞争结构。"
                "因为经营杠杆抬升，所以波动会增加。"
            ),
            source="sample",
        ),
    ]


def test_extract_author_profile_from_documents_returns_keywords_topics_and_patterns():
    profile = extract_author_profile_from_documents(_sample_documents())

    assert profile.author == "黄彦臻"
    assert profile.document_count == 2
    assert profile.keywords
    assert profile.focus_topics
    assert profile.reasoning_patterns
    assert any("因为" in item.pattern for item in profile.reasoning_patterns)


def test_build_author_profile_supports_author_filter(monkeypatch):
    docs = _sample_documents()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "expert_digest.knowledge.author_profile.get_documents_by_author",
        lambda db_path, author: (
            captured.update({"db_path": Path(db_path), "author": author}) or docs
        ),
    )

    profile = build_author_profile(
        db_path=Path("data/processed/zhihu_huang.sqlite3"),
        author="黄彦臻",
    )

    assert profile.document_count == 2
    assert captured["author"] == "黄彦臻"


def test_build_author_profile_raises_when_no_documents(monkeypatch):
    monkeypatch.setattr(
        "expert_digest.knowledge.author_profile.list_documents",
        lambda *_a, **_k: [],
    )

    with pytest.raises(ValueError, match="no documents available"):
        build_author_profile(
            db_path=Path("data/processed/zhihu_huang.sqlite3"),
            author=None,
        )
