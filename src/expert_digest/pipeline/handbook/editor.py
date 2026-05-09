"""Global coherence pass: assemble chapters into a consistent handbook."""

from __future__ import annotations

from expert_digest.pipeline.llm import require_fast_client
from expert_digest.pipeline.state import DigestState

_EDITOR_SYSTEM_PROMPT = """\
你是一位图书编辑。你的任务是将若干章节合并为一本完整的学习手册。

要求：
1. 在开头添加一个简短的引言，说明本手册的学习目标
2. 确保全书的标题层级一致（# 手册标题 > ## 章节标题 > ### 小节标题）
3. 在章节之间添加平滑的过渡段落（1-2句话）
4. 消除跨章节的重复内容
5. 统一术语
6. 在末尾添加一个简短的结语

输出完整的 markdown 格式手册。不要输出任何多余文本。"""


def _build_editor_prompt(chapters: list[dict], author: str, themes: list) -> str:
    """Build a prompt with all chapters for the coherence LLM call."""
    parts: list[str] = [
        f"# {author} 的学习手册\n\n",
        f"本手册基于对 {author} 的文章分析整理而成。"
        f"共 {len(chapters)} 个章节。\n\n",
    ]

    for i, chapter in enumerate(chapters, 1):
        title = chapter.get("title", f"第{i}章")
        content = chapter.get("content", "")
        parts.append(f"---\n## {title}\n{content}\n")

    return "\n".join(parts)


def run_coherence_pass(state: DigestState) -> dict:
    """Apply global consistency editing across all chapters."""
    chapters = state.get("chapters", [])
    if not chapters:
        return {"handbook_markdown": ""}

    author = state.get("author", "作者")
    themes = state.get("themes", [])
    chapter_dicts = [
        {"title": c.title, "content": c.content} for c in chapters
    ]

    llm = require_fast_client()
    user_prompt = _build_editor_prompt(chapter_dicts, author, themes)
    raw = llm.generate(
        system_prompt=_EDITOR_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    handbook_md = raw.strip()
    return {"handbook_markdown": handbook_md}
