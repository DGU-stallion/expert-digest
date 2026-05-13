"""Tests for LLM-based wiki analyzer."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from expert_digest.wiki.analyzer import analyze_document


@pytest.fixture
def mock_llm():
    fake_raw = """\
{
  "summary": "分析了泡泡玛特的IP运营能力。",
  "key_claims": [
    "泡泡玛特的核心能力是IP运营。",
    "估值不能只看玩具销售。"
  ],
  "concepts": ["泡泡玛特", "IP运营", "角色资产", "估值模型"],
  "topics": ["潮玩行业", "IP经济"]
}"""
    with patch(
        "expert_digest.wiki.analyzer.require_fast_client"
    ) as mock_factory:
        mock_client = mock_factory.return_value
        mock_client.generate.return_value = fake_raw
        yield mock_client


def test_analyze_document_returns_source_analysis(mock_llm):
    doc = {
        "id": "doc1",
        "title": "泡泡玛特复盘",
        "content": "泡泡玛特的核心能力是 IP 运营。",
        "author": "黄彦臻",
        "url": "https://example.com/popmart",
    }
    result = analyze_document(doc)

    assert result.source_id == "doc1"
    assert result.source_title == "泡泡玛特复盘"
    assert "IP运营" in result.concepts
    assert len(result.key_claims) == 2
    assert result.topics == ["潮玩行业", "IP经济"]
    assert result.confidence == "high"
    assert result.evidence_span_ids == []


def test_analyze_document_empty_content():
    doc = {
        "id": "empty",
        "title": "空文章",
        "content": "",
        "author": "test",
        "url": None,
    }
    result = analyze_document(doc)

    assert result.confidence == "low"
    assert result.key_claims == []
    assert result.concepts == []


def test_analyze_document_truncates_long_content():
    long = "word " * 4000
    doc = {
        "id": "long",
        "title": "长文",
        "content": long,
        "author": "test",
        "url": None,
    }
    with patch(
        "expert_digest.wiki.analyzer.require_fast_client"
    ) as mock_factory:
        mock_client = mock_factory.return_value
        mock_client.generate.return_value = '{"summary":"ok","key_claims":[],"concepts":[],"topics":[]}'
        result = analyze_document(doc)

    sent_text = mock_client.generate.call_args[1]["user_prompt"]
    assert len(sent_text) < 7000


def test_analyze_document_handles_malformed_json():
    doc = {
        "id": "bad",
        "title": "坏JSON",
        "content": "一些内容。",
        "author": "test",
        "url": None,
    }
    with patch(
        "expert_digest.wiki.analyzer.require_fast_client"
    ) as mock_factory:
        mock_client = mock_factory.return_value
        mock_client.generate.return_value = "not json at all"
        result = analyze_document(doc)

    assert result.confidence == "low"
    assert result.key_claims == []
    assert result.concepts == []
    assert result.topics == []


def test_analyze_document_strips_markdown_fences():
    doc = {
        "id": "md",
        "title": "带fence的",
        "content": "内容。",
        "author": "test",
        "url": None,
    }
    with patch(
        "expert_digest.wiki.analyzer.require_fast_client"
    ) as mock_factory:
        mock_client = mock_factory.return_value
        mock_client.generate.return_value = """```json
{"summary":"s","key_claims":["c1"],"concepts":["x"],"topics":["y"]}
```"""
        result = analyze_document(doc)

    assert result.concepts == ["x"]
    assert result.key_claims == ["c1"]
