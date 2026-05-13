"""LLM-driven source analysis for wiki ingest.

Replaces the old rule-based analyzer with direct LLM extraction
of summary, key claims, concepts, and topics from full documents.
"""

from __future__ import annotations

import json
import re
from typing import Any

from expert_digest.pipeline.llm import require_fast_client
from expert_digest.wiki.models import SourceAnalysis

_ANALYZER_SYSTEM_PROMPT = """\
你是一位专业的内容分析师。你的任务是从一篇文章中提取结构化信息。

请提取以下内容：
1. **摘要**（2-3句话）：概括文章的核心内容
2. **关键论断**（3-5条）：作者的核心观点或判断，附上原文中的直接证据片段
3. **概念**（3-8个）：文章中反复出现或至关重要的专业术语/概念
4. **主题**（2-4个）：文章所属的主题领域

必须只输出 JSON 对象，不要输出任何多余文本。JSON 格式：
{
  "summary": "文章摘要",
  "key_claims": ["论断1（证据：原文片段）", "论断2（证据：原文片段）"],
  "concepts": ["概念1", "概念2"],
  "topics": ["主题1", "主题2"]
}"""


def _build_prompt(title: str, content: str) -> str:
    max_chars = 6000
    if len(content) > max_chars:
        content = content[:max_chars] + "\n\n...（文章过长，已截断）"
    return f"## 标题\n{title}\n\n## 正文\n{content}"


def analyze_document(document: dict[str, Any]) -> SourceAnalysis:
    """Extract structured info from a full document via LLM.

    Args:
        document: dict with keys ``id``, ``title``, ``content``, ``author``, ``url``.

    Returns:
        A ``SourceAnalysis`` dataclass instance.
    """
    doc_id = document.get("id", "")
    title = document.get("title", "").strip()
    content = document.get("content", "").strip()
    author = document.get("author", "未知")
    url = document.get("url")

    if not content:
        return SourceAnalysis(
            source_id=doc_id,
            source_title=title or "未命名",
            author=author,
            url=url,
            summary="",
            key_claims=[],
            concepts=[],
            topics=[],
            evidence_span_ids=[],
            confidence="low",
        )

    llm = require_fast_client()
    user_prompt = _build_prompt(title, content)
    raw = llm.generate(
        system_prompt=_ANALYZER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    parsed = _parse_json(raw)
    summary = str(parsed.get("summary", "")).strip()
    key_claims = _ensure_str_list(parsed.get("key_claims", []))
    concepts = _ensure_str_list(parsed.get("concepts", []))
    topics = _ensure_str_list(parsed.get("topics", []))
    confidence = "high" if key_claims else "low"

    return SourceAnalysis(
        source_id=doc_id,
        source_title=title or "未命名",
        author=author,
        url=url,
        summary=summary or f"《{title}》的核心内容摘要。",
        key_claims=key_claims,
        concepts=concepts,
        topics=topics,
        evidence_span_ids=[],
        confidence=confidence,
    )


def _parse_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _ensure_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if isinstance(item, str) and item.strip()]
