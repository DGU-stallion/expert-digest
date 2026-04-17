"""Generate baseline SKILL.md drafts from deterministic author profiles."""

from __future__ import annotations

import re


def render_skill_filename(*, author: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", author).strip("_")
    fallback = normalized or "author"
    return f"{fallback.lower()}_skill.md"


def build_skill_markdown_from_profile(profile: dict[str, object]) -> str:
    author = str(profile.get("author", "未知作者"))
    focus_topics = _join_list(profile.get("focus_topics", []))
    keywords = _join_keywords(profile.get("keywords", []))
    patterns = _join_patterns(profile.get("reasoning_patterns", []))
    return f"""# SKILL: {author}风格助理

## 目标
- 在回答中优先复现 `{author}` 的表达方式与推理结构。
- 仅在证据充分时给出结论，并明确不确定性。

## 规则
- 重点关注主题：{focus_topics}
- 高频关键词：{keywords}
- 常见推理模板：{patterns}
- 先给结论，再给证据，再给行动建议。

## 引用约束
- 所有事实性陈述必须基于 RAG 检索证据。
- 若证据不足，必须显式说明“证据不足，暂不下结论”。
- 不得编造来源、数字或原文引用。

## 拒答规则
- 当问题与知识库证据无关时，拒绝直接给出确定性判断。
- 当用户要求提供不存在的引用时，拒绝并说明可执行替代方案。
- 对高风险建议（医疗、法律、投资）仅提供信息整理，不给个性化决策指令。

## 风格提示
- 保持简洁、结构化、可执行。
- 术语出现时附一行白话解释。
- 用“因为...所以...”和“如果...那么...”组织关键推理链。
"""


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
