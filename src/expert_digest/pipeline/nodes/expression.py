"""Expression DNA analysis: LLM-driven stylistic and intellectual profiling."""

from __future__ import annotations

import json
import re

from expert_digest.pipeline.llm import require_fast_client
from expert_digest.pipeline.state import DigestState, ExpressionDNA

_EXPRESSION_SYSTEM_PROMPT = """\
你是一位文体学家和知性传记作者。你的任务是从作者的文章中分析其表达风格和智识特征。

请分析以下维度：
1. **表达DNA**：句式结构特点（如"结论先行"）、高频短语、确定性表达频谱、引用/举例习惯
2. **智识谱系**：作者的思想渊源、受到谁/什么流派影响、核心认知框架来源
3. **关键判断**：作者在重要议题上做出过的标志性判断或立场（3-5个）

必须只输出 JSON 对象，不要输出任何多余文本。JSON 格式：
{
  "sentence_patterns": ["句式特点1", "句式特点2"],
  "high_frequency_phrases": ["高频短语1", "高频短语2"],
  "certainty_spectrum": ["确定性表达方式1", "确定性表达方式2"],
  "citation_habits": "引用习惯描述",
  "intellectual_genealogy": "智识谱系描述",
  "key_decisions": [
    {"topic": "议题", "position": "立场", "evidence": "依据摘要"}
  ]
}"""


def _build_prompt(wiki_pages: list[dict], documents: list[dict]) -> str:
    """Build prompt from wiki source pages + full doc excerpts."""
    parts: list[str] = []
    author = documents[0].get("author", "未知") if documents else "未知"

    # Prefer wiki source pages for richer stylistic context
    sources = [p for p in wiki_pages if p.get("page_type") == "source"]
    if sources:
        for s in sources[:20]:
            title = s.get("title", "")
            body = s.get("body", "")
            excerpt = body[:500] if len(body) > 500 else body
            parts.append(f"## {title}\n{excerpt}")

    # Also sample full documents for deeper analysis
    for doc in documents[:8]:
        title = doc.get("title", "").strip()
        content = doc.get("content", "").strip()
        excerpt = content[:800] if len(content) > 800 else content
        if excerpt:
            parts.append(f"## {title}\n{excerpt}")

    return (
        f"以下是作者 {author} 的文章素材，请分析其表达特征。\n\n"
        + "\n\n".join(parts)
    )


def run_analyze_expression(state: DigestState) -> dict:
    """Analyze the author's expression style and communication DNA."""
    documents = state.get("documents", [])
    if not documents:
        return {
            "expression_dna": None,
            "intellectual_genealogy": "",
            "key_decisions": [],
        }

    wiki_pages = state.get("wiki_pages", [])
    llm = require_fast_client()
    user_prompt = _build_prompt(wiki_pages, documents)
    raw = llm.generate(
        system_prompt=_EXPRESSION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    parsed = _parse_expression_json(raw)

    expression_dna = ExpressionDNA(
        sentence_patterns=_ensure_str_list(parsed.get("sentence_patterns", [])),
        high_frequency_phrases=_ensure_str_list(
            parsed.get("high_frequency_phrases", [])
        ),
        certainty_spectrum=_ensure_str_list(parsed.get("certainty_spectrum", [])),
        citation_habits=str(parsed.get("citation_habits", "")),
    )
    intellectual_genealogy = str(parsed.get("intellectual_genealogy", ""))
    key_decisions = parsed.get("key_decisions", [])
    if not isinstance(key_decisions, list):
        key_decisions = []

    return {
        "expression_dna": expression_dna,
        "intellectual_genealogy": intellectual_genealogy,
        "key_decisions": [dict(d) for d in key_decisions if isinstance(d, dict)],
    }


def _parse_expression_json(raw: str) -> dict:
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
    return [
        str(item).strip()
        for item in value
        if isinstance(item, str) and item.strip()
    ]
