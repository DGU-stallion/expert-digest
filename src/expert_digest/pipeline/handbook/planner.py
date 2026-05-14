"""Chapter planning node: LLM-driven handbook outline generation."""

from __future__ import annotations

import json
import re

from expert_digest.pipeline.llm import require_reasoning_client
from expert_digest.pipeline.state import ChapterPlan, DigestState

_PLANNER_SYSTEM_PROMPT = """\
你是一位教育内容设计师。你的任务是根据作者的文章聚类分析结果，规划一本系统性的学习手册的章节结构。

要求：
1. 规划 5-10 个章节，形成从基础认知到深度应用的学习路径
2. 每个章节聚焦一个聚类主题领域，展开深度叙述
3. 章节之间要有逻辑递进关系，前章为后章打基础
4. 大聚类（chunk 数量多的）应分配更多章节空间
5. 忽略零散的小主题，聚焦核心聚类

必须只输出 JSON 数组，不要输出任何多余文本。JSON 格式：
[
  {
    "title": "章节标题",
    "purpose": "本章学习目的（2-3句话，说明本章在整个学习路径中的定位）",
    "target_themes": ["关联的聚类标签1"],
    "estimated_sections": 4
  }
]"""


def _build_planner_prompt(state: DigestState) -> str:
    """Build the user prompt for chapter planning from cluster results and analysis."""
    clusters = state.get("topic_clusters", [])
    themes = state.get("themes", [])
    concepts = state.get("concepts", [])

    parts: list[str] = []

    if clusters:
        parts.append(f"## 文章聚类结果（共 {len(clusters)} 个主要聚类）\n")
        parts.append("以下是通过社区检测算法从所有文章片段中发现的聚类主题，按规模降序排列：\n")
        sorted_clusters = sorted(
            clusters, key=lambda c: c.get("chunk_count", 0), reverse=True
        )
        for i, c in enumerate(sorted_clusters, 1):
            label = c.get("label", f"聚类{i}")
            size = c.get("chunk_count", 0)
            rep_docs = c.get("representative_documents", [])
            doc_titles = [d.get("title", "") for d in rep_docs[:3] if d.get("title")]
            parts.append(
                f"{i}. **{label}** (规模: {size} 个片段)"
                + (f" — 代表性文章: {', '.join(doc_titles)}" if doc_titles else "")
            )

    if themes:
        parts.append("\n## LLM 提取的主题（补充参考）")
        for t in themes[:5]:
            parts.append(f"- {t.label}：{t.summary}")

    if concepts:
        parts.append("\n## 关键概念")
        parts.append("、".join(concepts[:12]))

    return "\n".join(parts)


def run_plan_chapters(state: DigestState) -> dict:
    """Plan handbook chapters based on analyzed themes and concepts."""
    themes = state.get("themes", [])
    if not themes:
        return {"chapter_plan": []}

    print("  [handbook] plan_chapters: LLM outline generation...")
    llm = require_reasoning_client()
    user_prompt = _build_planner_prompt(state)
    raw = llm.generate(
        system_prompt=_PLANNER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    plans = _parse_chapter_plan(raw)
    print(f"  [handbook] plan_chapters: {len(plans)} chapters planned")
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
