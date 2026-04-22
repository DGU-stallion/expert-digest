"""Generate LLM-only structured SKILL.md drafts from author profiles."""

from __future__ import annotations

import json
import re
from typing import Protocol


class SkillWriterLLMClient(Protocol):
    """Protocol for skill generation LLM client."""

    def generate(self, *, system_prompt: str, user_prompt: str) -> str: ...


def render_skill_filename(*, author: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", author).strip("_")
    fallback = normalized or "author"
    return f"{fallback.lower()}_skill.md"


def build_skill_markdown_from_profile(
    profile: dict[str, object],
    *,
    llm_client: SkillWriterLLMClient,
) -> str:
    author = str(profile.get("author", "未知作者"))
    sections = _generate_structured_sections(profile=profile, llm_client=llm_client)
    return f"""# SKILL: {author}风格助理

## 风格原则
{_to_bullets(sections["style_principles"])}

## 回答流程
{_to_bullets(sections["response_flow"])}

## 风险守则
{_to_bullets(sections["risk_guardrails"])}

## 拒答策略
{_to_bullets(sections["refusal_policy"])}
"""


def _generate_structured_sections(
    *,
    profile: dict[str, object],
    llm_client: SkillWriterLLMClient,
) -> dict[str, list[str]]:
    author = str(profile.get("author", "未知作者"))
    focus_topics = _join_list(profile.get("focus_topics", []))
    keywords = _join_keywords(profile.get("keywords", []))
    patterns = _join_patterns(profile.get("reasoning_patterns", []))

    system_prompt = (
        "你是资深知识工程编辑。请根据作者画像生成 SKILL.md 的结构化草稿内容。"
        "必须只输出 JSON 对象，不要输出任何多余文本。"
    )
    user_prompt = (
        f"作者: {author}\n"
        f"重点主题: {focus_topics}\n"
        f"高频关键词: {keywords}\n"
        f"推理模式: {patterns}\n\n"
        "输出 JSON 对象，必须包含以下 4 个字段，且每个字段是长度 3-6 的字符串数组：\n"
        '- "style_principles"\n'
        '- "response_flow"\n'
        '- "risk_guardrails"\n'
        '- "refusal_policy"\n'
        "所有条目必须是可执行、可检查的简短中文句子。"
    )
    raw = llm_client.generate(system_prompt=system_prompt, user_prompt=user_prompt)
    return _parse_structured_sections(raw)


def _parse_structured_sections(raw: str) -> dict[str, list[str]]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("skill_generation_failed") from error
    if not isinstance(parsed, dict):
        raise ValueError("skill_generation_failed")

    result: dict[str, list[str]] = {}
    for field in (
        "style_principles",
        "response_flow",
        "risk_guardrails",
        "refusal_policy",
    ):
        value = parsed.get(field)
        if not isinstance(value, list):
            raise ValueError("skill_generation_failed")
        lines = [str(item).strip() for item in value if str(item).strip()]
        if not lines:
            raise ValueError("skill_generation_failed")
        result[field] = lines
    return result


def _to_bullets(lines: list[str]) -> str:
    return "\n".join(f"- {line}" for line in lines)


def _join_list(items: object) -> str:
    if not isinstance(items, list) or not items:
        return "（暂无）"
    return "、".join(str(item) for item in items[:6])


def _join_keywords(items: object) -> str:
    if not isinstance(items, list) or not items:
        return "（暂无）"
    labels: list[str] = []
    for item in items[:8]:
        if isinstance(item, dict):
            labels.append(str(item.get("keyword", "")))
    labels = [item for item in labels if item]
    return "、".join(labels) if labels else "（暂无）"


def _join_patterns(items: object) -> str:
    if not isinstance(items, list) or not items:
        return "（暂无）"
    labels: list[str] = []
    for item in items[:5]:
        if isinstance(item, dict):
            labels.append(str(item.get("pattern", "")))
    labels = [item for item in labels if item]
    return "、".join(labels) if labels else "（暂无）"
