"""Content analysis node: LLM-driven theme, concept, and pattern extraction."""

from __future__ import annotations

import json
import re

from expert_digest.pipeline.llm import require_reasoning_client
from expert_digest.pipeline.state import DigestState, Theme

_ANALYZER_SYSTEM_PROMPT = """\
你是一位专业的内容分析师。你的任务是从作者的文章中提炼出核心主题、关键概念和思维模式。

要求：
1. 找出 3-6 个核心主题：标签、摘要、相关文章标题、相关概念
2. 列出 8-15 个高频核心概念/术语
3. 识别 3-8 个典型的思维模式/推理方式

必须只输出 JSON 对象，不要输出任何多余文本。JSON 格式：
{
  "themes": [
    {
      "label": "主题标签",
      "summary": "主题摘要",
      "source_titles": ["文章标题1", "文章标题2"],
      "related_concepts": ["概念1", "概念2"]
    }
  ],
  "concepts": ["概念1", "概念2"],
  "thinking_patterns": ["模式1", "模式2"]
}"""


def _build_wiki_prompt(wiki_pages: list[dict], documents: list[dict]) -> str:
    """Build prompt from wiki topic/concept pages, falling back to documents."""
    parts: list[str] = []
    author = documents[0].get("author", "未知") if documents else "未知"

    topics = [p for p in wiki_pages if p.get("page_type") == "topic"]
    concepts = [p for p in wiki_pages if p.get("page_type") == "concept"]

    if topics:
        parts.append(f"## Wiki 主题（共 {len(topics)} 个）")
        for t in topics[:10]:
            title = t.get("title", "")
            body = t.get("body", "")
            excerpt = body[:500] if len(body) > 500 else body
            parts.append(f"### {title}\n{excerpt}")

    if concepts:
        parts.append(f"\n## Wiki 概念（共 {len(concepts)} 个）")
        for c in concepts[:15]:
            title = c.get("title", "")
            body = c.get("body", "")
            excerpt = body[:300] if len(body) > 300 else body
            parts.append(f"### {title}\n{excerpt}")

    # If no wiki data, fall back to document excerpts
    if not topics and not concepts:
        sampled = documents[:20]
        for doc in sampled:
            title = doc.get("title", "").strip()
            content = doc.get("content", "").strip()
            excerpt = content[:300] if len(content) > 300 else content
            parts.append(f"## {title}\n{excerpt}")

    return (
        f"以下是作者 {author} 的内容分析素材。"
        + ("\n\n" + "\n\n".join(parts))
    )


def run_analyze_content(state: DigestState) -> dict:
    """Analyze content to extract themes, concepts, and thinking patterns."""
    documents = state.get("documents", [])
    if not documents:
        return {"themes": [], "concepts": [], "thinking_patterns": []}

    wiki_pages = state.get("wiki_pages", [])
    llm = require_reasoning_client()
    user_prompt = _build_wiki_prompt(wiki_pages, documents)
    raw = llm.generate(
        system_prompt=_ANALYZER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    parsed = _parse_analysis_json(raw)

    themes = _build_themes(parsed.get("themes", []), documents)
    concepts = parsed.get("concepts", [])
    thinking_patterns = parsed.get("thinking_patterns", [])

    if not isinstance(concepts, list):
        concepts = []
    if not isinstance(thinking_patterns, list):
        thinking_patterns = []

    return {
        "themes": themes,
        "concepts": [str(c) for c in concepts if isinstance(c, str) and c.strip()],
        "thinking_patterns": [
            str(p) for p in thinking_patterns if isinstance(p, str) and p.strip()
        ],
    }


def _build_themes(
    raw_themes: list[dict],
    documents: list[dict],
) -> list[Theme]:
    """Convert raw theme dicts from LLM into Theme dataclass instances."""
    title_to_id = {doc.get("title", ""): doc.get("id", "") for doc in documents}
    result: list[Theme] = []
    for item in raw_themes[:6]:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        summary = str(item.get("summary", "")).strip()
        if not label or not summary:
            continue
        source_titles = item.get("source_titles", [])
        if isinstance(source_titles, list):
            source_ids = [
                title_to_id.get(t, "") for t in source_titles if isinstance(t, str)
            ]
            source_ids = [sid for sid in source_ids if sid]
        else:
            source_ids = []
        related_raw = item.get("related_concepts", [])
        related = (
            [str(c) for c in related_raw if isinstance(c, str) and c.strip()]
            if isinstance(related_raw, list)
            else []
        )
        result.append(
            Theme(
                label=label,
                summary=summary,
                source_document_ids=source_ids,
                related_concepts=related,
            )
        )
    return result


def _parse_analysis_json(raw: str) -> dict:
    """Parse LLM JSON response, stripping markdown fences if present."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    key_map = {
        "core_themes": "themes",
        "key_concepts": "concepts",
        "thinking_patterns": "thinking_patterns",
        "thinking_pattern": "thinking_patterns",
    }
    for old_key, new_key in key_map.items():
        if old_key != new_key and old_key in parsed and new_key not in parsed:
            parsed[new_key] = parsed.pop(old_key)
    return parsed
