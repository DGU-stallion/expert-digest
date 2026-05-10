"""Global coherence pass: 3-stage editing to assemble chapters into a consistent handbook."""

from __future__ import annotations

from expert_digest.pipeline.llm import require_reasoning_client
from expert_digest.pipeline.state import DigestState

# ── Stage 1: Dedup + terminology ──

_DEDUP_SYSTEM_PROMPT = """\
你是一位图书编辑。你的任务是审查学习手册的全部章节，消除重复内容并统一术语。

要求：
1. 识别跨章节的重复论述（尤其是相同的案例、论据），标记并删除重复
2. 统一全书的术语表达（例如全篇统一用"流动性周期"而非混用"流动性循环"）
3. 确保每个案例和论点只在最优位置出现一次
4. 保留所有章节的内容，只做去重和术语替换

输出处理后的完整 markdown 手册。不要输出任何多余文本。"""

# ── Stage 2: Flow + logic ──

_FLOW_SYSTEM_PROMPT = """\
你是一位资深内容编辑。你的任务是优化学习手册的章节逻辑流。

要求：
1. 检查每个章节内部的论点推进是否逻辑连贯
2. 必要时调整小节段落顺序，使论证层层递进
3. 在知识点衔接处添加说明性连接句
4. 确保每个章节的小节都能支撑该章的学习目的
5. 保留全部内容，只做结构优化

输出优化后的完整 markdown 手册。不要输出任何多余文本。"""

# ── Stage 3: Polish + transitions ──

_POLISH_SYSTEM_PROMPT = """\
你是一位终审编辑。你的任务是完成学习手册的最终润色。

要求：
1. 在开头添加一个简短的引言（2-3段），说明本手册的学习目标和读者收获
2. 在章节之间添加自然的过渡段落（2-3句话），建立知识衔接
3. 确保全文标题层级一致：# 手册标题 > ## 章节标题 > ### 小节标题
4. 统一语气风格：清晰、专业、有洞见
5. 在末尾添加结语，总结学习路径并给出继续深入的方向

输出最终版本的完整 markdown 手册。不要输出任何多余文本。"""


def _assemble_raw(chapters: list[dict], author: str) -> str:
    """Assemble chapter dicts into raw markdown for editing."""
    parts: list[str] = [
        f"# {author} 的学习手册\n\n",
        f"本手册基于对 {author} 的文章的系统化整理。共 {len(chapters)} 个章节。\n\n",
    ]
    for i, ch in enumerate(chapters, 1):
        title = ch.get("title", f"第{i}章")
        content = ch.get("content", "")
        parts.append(f"\n---\n\n## {title}\n\n{content}\n")
    return "\n".join(parts)


def _edit_pass(
    text: str, system_prompt: str, label: str
) -> str:
    """Run a single editing pass via LLM. Returns original text on error."""
    try:
        llm = require_reasoning_client()
        result = llm.generate(
            system_prompt=system_prompt,
            user_prompt=text,
        )
        return result.strip() or text
    except Exception:
        return text


def run_coherence_pass(state: DigestState) -> dict:
    """Apply 3-stage global editing across all chapters."""
    chapters = state.get("chapters", [])
    if not chapters:
        return {"handbook_markdown": ""}

    author = state.get("author", "作者")
    chapter_dicts = [
        {"title": c.title, "content": c.content} for c in chapters
    ]

    raw = _assemble_raw(chapter_dicts, author)

    # Stage 1: Deduplicate + unify terminology
    raw = _edit_pass(raw, _DEDUP_SYSTEM_PROMPT, "dedup")

    # Stage 2: Fix logical flow + reorder sections
    raw = _edit_pass(raw, _FLOW_SYSTEM_PROMPT, "flow")

    # Stage 3: Add introduction, transitions, conclusion, polish
    raw = _edit_pass(raw, _POLISH_SYSTEM_PROMPT, "polish")

    return {"handbook_markdown": raw}
