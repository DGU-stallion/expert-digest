"""Agentic Protocol design for SKILL.md via LLM."""

from __future__ import annotations

import json
import re

from expert_digest.pipeline.llm import require_fast_client
from expert_digest.pipeline.state import DigestState

_PROTOCOL_SYSTEM_PROMPT = """\
你是一位认知架构设计师。你的任务是根据作者的思维体系，设计一套AI使用其知识时的"回答工作流"（Agentic Protocol）。

已有数据：
- 心智模型：作者的核心分析框架
- 决策启发式：作者做判断时的经验法则
- 核心主题：作者关注的领域

请设计一个三步工作流：
1. **问题分类**：列出用户问题类型及对应的处理策略（表格形式）
2. **研究维度**：根据心智模型推导出分析问题的维度（按领域分组）
3. **回答框架**：作者式的回答输出结构

必须只输出完整的 markdown 格式内容，不要输出任何多余文本。"""


def _build_protocol_prompt(state: DigestState) -> str:
    models = state.get("mental_models", [])
    heuristics = state.get("decision_heuristics", [])
    themes = state.get("themes", [])

    parts: list[str] = []
    if models:
        parts.append("## 心智模型")
        for m in models[:7]:
            parts.append(f"- {m.name}：{m.summary}")
    if heuristics:
        parts.append("## 决策启发式")
        for h in heuristics[:8]:
            parts.append(f"- {h}")
    if themes:
        parts.append("## 核心主题")
        for t in themes[:6]:
            parts.append(f"- {t.label}：{t.summary}")

    return "\n".join(parts)


def run_design_protocol(state: DigestState) -> dict:
    """Design Agentic Protocol based on mental models and themes."""
    models = state.get("mental_models", [])
    if not models:
        return {"protocol_steps": ""}

    llm = require_fast_client()
    user_prompt = _build_protocol_prompt(state)
    raw = llm.generate(
        system_prompt=_PROTOCOL_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    protocol_steps = raw.strip()
    return {"protocol_steps": protocol_steps}
