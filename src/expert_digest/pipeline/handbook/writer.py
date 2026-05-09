"""Chapter drafting node with LLM writing and rewrite feedback support."""

from __future__ import annotations

from expert_digest.pipeline.llm import require_fast_client
from expert_digest.pipeline.state import ChapterDraft, DigestState

_WRITER_SYSTEM_PROMPT = """\
你是一位教育内容作家。你的任务是根据作者的文章原文，为学习手册撰写一个章节。

要求：
1. 基于提供的原文材料撰写，不要编造内容和引用
2. 用中文撰写，风格清晰、结构化
3. 适当引用作者的原话或案例作为论据
4. 每个小节用 ## 标题 分隔
5. 内容要有深度，展示作者的独特见解而非泛泛之谈

必须只输出 markdown 格式的章节内容，不要输出任何多余文本。"""


def _build_writer_prompt(
    plan: ChapterPlan,
    documents: list[dict],
    review_feedback: list[str] | None = None,
) -> str:
    """Build the user prompt for a single chapter draft."""
    doc_lines: list[str] = []
    for doc in documents:
        title = doc.get("title", "").strip()
        content = doc.get("content", "").strip()
        excerpt = content[:1000] if len(content) > 1000 else content
        doc_lines.append(f"### {title}\n{excerpt}")

    prompt = (
        f"请撰写以下章节：\n\n"
        f"## 章节标题\n{plan.title}\n\n"
        f"## 学习目的\n{plan.purpose}\n\n"
        f"## 目标主题\n"
        + "\n".join(f"- {t}" for t in plan.target_themes)
        + "\n\n"
        f"## 参考原文材料\n"
        + "\n\n".join(doc_lines)
    )

    if review_feedback:
        prompt += (
            "\n\n## 上一轮评审意见\n"
            + "\n".join(f"- {issue}" for issue in review_feedback)
            + "\n\n请根据以上意见修改章节内容，确保这些问题已解决。"
        )

    return prompt


def _find_relevant_docs(
    plan: ChapterPlan,
    state: DigestState,
    max_docs: int = 5,
) -> list[dict]:
    """Find documents relevant to a chapter's target themes."""
    documents = state.get("documents", [])

    # Build theme label → source document IDs map
    theme_doc_ids: dict[str, list[str]] = {}
    for theme in state.get("themes", []):
        if theme.source_document_ids:
            theme_doc_ids[theme.label] = theme.source_document_ids

    # Collect doc IDs from the chapter's target themes
    doc_ids: set[str] = set()
    for theme_label in plan.target_themes:
        matched = theme_doc_ids.get(theme_label, [])
        doc_ids.update(matched)

    # Map IDs to actual docs
    doc_map = {d.get("id", ""): d for d in documents}
    relevant = [doc_map[did] for did in doc_ids if did in doc_map]

    # Fallback: return first N docs if no theme match
    if not relevant:
        return documents[:max_docs]

    return relevant[:max_docs]


def run_draft_chapters(state: DigestState) -> dict:
    """Draft all chapters, using review feedback if available on rewrite."""
    chapter_plan = state.get("chapter_plan", [])
    review_results = state.get("review_results", [])

    if not chapter_plan:
        return {}

    # Collect review feedback per chapter for rewrite pass
    feedback_map: dict[str, list[str]] = {}
    for rr in review_results:
        if rr.issues:
            feedback_map[rr.chapter_title] = rr.issues

    llm = require_fast_client()
    chapters: list[ChapterDraft] = []

    for plan in chapter_plan:
        relevant_docs = _find_relevant_docs(plan, state)
        feedback = feedback_map.get(plan.title, None)
        user_prompt = _build_writer_prompt(plan, relevant_docs, feedback)
        raw = llm.generate(
            system_prompt=_WRITER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        content = raw.strip()
        section_count = content.count("\n##")
        chapters.append(
            ChapterDraft(
                title=plan.title,
                content=content,
                section_count=section_count,
            )
        )

    return {"chapters": chapters}
