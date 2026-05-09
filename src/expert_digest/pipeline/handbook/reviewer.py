"""Chapter quality review node with conditional routing for rewrite loop."""

from __future__ import annotations

import json
import re

from expert_digest.pipeline.llm import require_fast_client
from expert_digest.pipeline.state import (
    ChapterDraft,
    DigestState,
    PipelineError,
    ReviewResult,
)

_REVIEWER_SYSTEM_PROMPT = """\
你是一位教育内容质量评审。你的任务是评估学习手册章节的质量。

评审维度：
1. 事实依据：内容是否基于原文材料，没有编造
2. 结构完整：是否有清晰的标题层次
3. 深度：是否展示了作者的独特见解而非泛泛之谈
4. 重复：是否与已有章节内容有明显重复

必须只输出 JSON 对象，不要输出任何多余文本。JSON 格式：
{
  "passed": true,
  "issues": []
}

如果评审不通过，在 issues 中列出具体问题。"""


def _build_reviewer_prompt(
    chapter: ChapterDraft,
    all_titles: list[str],
) -> str:
    other = [t for t in all_titles if t != chapter.title]
    return (
        f"请评审以下章节：\n\n"
        f"## 章节标题\n{chapter.title}\n\n"
        f"## 章节内容\n{chapter.content[:2000]}\n\n"
        f"## 其他章节标题（供重复性检查）\n"
        + "\n".join(f"- {t}" for t in other)
    )


def _parse_review(raw: str) -> dict:
    """Parse LLM JSON review response."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"passed": False, "issues": ["failed to parse review response"]}
    return parsed if isinstance(parsed, dict) else {"passed": False, "issues": ["invalid review format"]}


def run_review_chapters(state: DigestState) -> dict:
    """Review all drafted chapters for quality and completeness."""
    chapters = state.get("chapters", [])
    if not chapters:
        return {
            "review_results": [],
            "errors": [PipelineError(node="reviewer", message="no chapters to review")],
        }

    llm = require_fast_client()
    all_titles = [c.title for c in chapters]
    results: list[ReviewResult] = []

    for chapter in chapters:
        user_prompt = _build_reviewer_prompt(chapter, all_titles)
        raw = llm.generate(
            system_prompt=_REVIEWER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        parsed = _parse_review(raw)
        issues_raw = parsed.get("issues", [])
        issues = [str(i) for i in issues_raw if isinstance(i, str)] if isinstance(issues_raw, list) else []
        results.append(
            ReviewResult(
                chapter_title=chapter.title,
                passed=bool(parsed.get("passed", False)),
                issues=issues,
            )
        )

    return {"review_results": results}


def all_chapters_pass(state: DigestState) -> str:
    """Return 'rewrite' if any chapter failed, 'proceed' otherwise.

    The rewrite loop converges naturally because each iteration feeds
    review issues back to the writer. If the loop runs indefinitely
    the outer pipeline will still terminate through its own max_rounds gate.
    """
    review_results = state.get("review_results", [])
    if not review_results:
        return "proceed"
    failed = [r for r in review_results if not r.passed]
    return "rewrite" if failed else "proceed"
