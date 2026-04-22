"""LLM-driven multi-stage helpers for handbook chapter generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol


class BookPipelineLLMClient(Protocol):
    """Protocol for book generation LLM client."""

    def generate(self, *, system_prompt: str, user_prompt: str) -> str: ...


@dataclass(frozen=True)
class ChapterPlanItem:
    title: str
    objective: str


@dataclass(frozen=True)
class ChapterMarkdown:
    title: str
    objective: str
    body: str


class BookPipeline:
    """Build chapter plan and chapter drafts via LLM."""

    def __init__(self, *, llm_client: BookPipelineLLMClient) -> None:
        self._llm_client = llm_client

    def build_plan(self, *, sections: list[dict[str, str]]) -> list[ChapterPlanItem]:
        system_prompt = (
            "你是一本学习手册的编辑。请根据输入主题生成章节计划，"
            "只输出 JSON 数组，每项包含 title 和 objective。"
        )
        section_lines = []
        for index, section in enumerate(sections, start=1):
            section_lines.append(
                f"{index}. name={section.get('name', '')}; "
                f"summary={section.get('summary', '')}"
            )
        user_prompt = (
            "主题输入：\n"
            + ("\n".join(section_lines) if section_lines else "(无)")
            + "\n\n请返回 JSON 数组。"
        )
        raw = self._llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        ).strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError("chapter_plan_failed") from error
        if not isinstance(parsed, list) or not parsed:
            raise ValueError("chapter_plan_failed")

        plans: list[ChapterPlanItem] = []
        for item in parsed:
            if not isinstance(item, dict):
                raise ValueError("chapter_plan_failed")
            title = str(item.get("title", "")).strip()
            objective = str(item.get("objective", "")).strip()
            if not title or not objective:
                raise ValueError("chapter_plan_failed")
            plans.append(ChapterPlanItem(title=title, objective=objective))
        return plans

    def draft_chapter(
        self,
        *,
        plan_item: ChapterPlanItem,
        section: dict[str, str],
    ) -> str:
        system_prompt = (
            "你是学习手册作者。根据章节目标与证据摘要，"
            "输出结构化、可学习的章节正文。"
        )
        user_prompt = (
            f"title: {plan_item.title}\n"
            f"objective: {plan_item.objective}\n"
            f"theme_name: {section.get('name', '')}\n"
            f"theme_summary: {section.get('summary', '')}\n"
            "\n请输出章节正文，不要输出链接。"
        )
        body = self._llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        ).strip()
        if not body:
            raise ValueError("chapter_draft_failed")
        return body


def build_chapter_markdowns(
    *,
    pipeline: BookPipeline,
    sections: list[dict[str, str]],
) -> list[ChapterMarkdown]:
    plan_items = pipeline.build_plan(sections=sections)
    chapter_markdowns: list[ChapterMarkdown] = []
    for index, plan_item in enumerate(plan_items):
        section = sections[min(index, len(sections) - 1)] if sections else {}
        body = pipeline.draft_chapter(plan_item=plan_item, section=section)
        chapter_markdowns.append(
            ChapterMarkdown(
                title=plan_item.title,
                objective=plan_item.objective,
                body=body,
            )
        )
    return chapter_markdowns
