"""Chapter planning node: LLM-driven handbook outline generation."""

from __future__ import annotations

import json
import re

from expert_digest.pipeline.llm import require_fast_client
from expert_digest.pipeline.state import ChapterPlan, DigestState

_PLANNER_SYSTEM_PROMPT = """\
你是一位教育内容设计师。你的任务是根据作者的文章分析结果，规划一本系统性的学习手册的章节结构。

要求：
1. 规划 5-8 个章节，形成从入门到深入的学习路径
2. 每个章节聚焦一个核心主题领域
3. 章节之间要有逻辑递进关系
4. 确保覆盖所有核心主题

必须只输出 JSON 数组，不要输出任何多余文本。JSON 格式：
[
  {
    "title": "章节标题",
    "purpose": "本章学习目的",
    "target_themes": ["关联的主题标签1"],
    "estimated_sections": 4
  }
]"""


def _build_planner_prompt(state: DigestState) -> str:
    """Build the user prompt for chapter planning from analysis results."""
    themes = state.get("themes", [])
    concepts = state.get("concepts", [])
    patterns = state.get("thinking_patterns", [])

    parts: list[str] = []
    parts.append(f"作者共有 {len(themes)} 个核心主题、{len(concepts)} 个关键概念、{len(patterns)} 个思维模式。\n")

    if themes:
        parts.append("## 核心主题")
        for t in themes[:6]:
            parts.append(f"- {t.label}：{t.summary}")

    if concepts:
        parts.append("## 关键概念")
        for c in concepts[:15]:
            parts.append(f"- {c}")

    if patterns:
        parts.append("## 思维模式")
        for p in patterns[:8]:
            parts.append(f"- {p}")

    return "\n".join(parts)


def run_plan_chapters(state: DigestState) -> dict:
    """Plan handbook chapters based on analyzed themes and concepts."""
    themes = state.get("themes", [])
    if not themes:
        return {"chapter_plan": []}

    llm = require_fast_client()
    user_prompt = _build_planner_prompt(state)
    raw = llm.generate(
        system_prompt=_PLANNER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    plans = _parse_chapter_plan(raw)
    return {"chapter_plan": plans}


def _parse_chapter_plan(raw: str) -> list[ChapterPlan]:
    """Parse LLM JSON response into ChapterPlan list."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []

    result: list[ChapterPlan] = []
    for item in parsed[:8]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        purpose = str(item.get("purpose", "")).strip()
        if not title or not purpose:
            continue
        target_themes = item.get("target_themes", [])
        if not isinstance(target_themes, list):
            target_themes = []
        estimated = item.get("estimated_sections", 3)
        if not isinstance(estimated, int):
            estimated = 3
        result.append(
            ChapterPlan(
                title=title,
                purpose=purpose,
                target_themes=[
                    str(t) for t in target_themes if isinstance(t, str) and t.strip()
                ],
                estimated_sections=max(1, min(10, estimated)),
            )
        )
    return result
