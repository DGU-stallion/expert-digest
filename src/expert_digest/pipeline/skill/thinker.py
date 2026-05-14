"""Mental model extraction from analysis results + original text via LLM."""

from __future__ import annotations

import json
import re

from expert_digest.pipeline.llm import require_reasoning_client
from expert_digest.pipeline.state import DigestState, MentalModel

_THINKER_SYSTEM_PROMPT = """\
你是一位认知科学家。你的任务是从作者的文章分析结果和原文中提炼其思维体系。

已有分析数据：
- 核心主题（themes）：作者的核心思想领域
- 关键概念（concepts）：作者常用的专业术语
- 思维模式（thinking_patterns）：作者典型的推理方式
- 表达DNA（expression_dna）：作者的表达风格特征
- 原文摘录：作者文章片段，作为证据源

请提炼：
1. **心智模型**（3-7个）：作者反复使用的分析框架/思维工具，每个模型需包含名称、一句话概括、证据片段、应用场景、局限性
2. **决策启发式**（3-8条）：作者在做判断时依赖的经验法则
3. **价值观与反模式**：作者的核心理念追求 vs 明确反对的做法
4. **诚实边界**（3-8条）：作者坦诚自己能力/知识边界的表述

必须只输出 JSON 对象，不要输出任何多余文本。JSON 格式：
{
  "mental_models": [
    {
      "name": "模型名称",
      "summary": "一句话概括",
      "evidence_snippet": "从原文中提取的真实证据片段",
      "application": "应用场景",
      "limitation": "局限性"
    }
  ],
  "decision_heuristics": ["启发式1", "启发式2"],
  "values_antipatterns": {
    "pursues": ["追求1", "追求2"],
    "opposes": ["反对1", "反对2"],
    "tensions": ["内在张力1"]
  },
  "honest_boundaries": ["边界1", "边界2"]
}"""


def _build_thinker_prompt(state: DigestState) -> str:
    themes = state.get("themes", [])
    concepts = state.get("concepts", [])
    patterns = state.get("thinking_patterns", [])
    dna = state.get("expression_dna")
    documents = state.get("documents", [])

    parts: list[str] = []
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
    if dna:
        parts.append(f"## 表达DNA\n句式特点：{dna.sentence_patterns}")
        parts.append(f"高频短语：{dna.high_frequency_phrases}")

    # Include original text excerpts as evidence source for mental models
    if documents:
        parts.append("## 原文摘录（从中提取心智模型的证据片段）")
        for doc in documents[:10]:
            title = doc.get("title", "").strip()
            content = doc.get("content", "").strip()
            excerpt = content[:600] if len(content) > 600 else content
            parts.append(f"### {title}\n{excerpt}")

    return "\n".join(parts)


def run_extract_mental_models(state: DigestState) -> dict:
    """Extract mental models, heuristics, values, and boundaries from analysis + docs."""
    themes = state.get("themes", [])
    if not themes:
        return {
            "mental_models": [],
            "decision_heuristics": [],
            "values_antipatterns": {},
            "honest_boundaries": [],
        }

    print("  [skill] extract_mental_models: LLM analysis extraction...")
    llm = require_reasoning_client()
    user_prompt = _build_thinker_prompt(state)
    raw = llm.generate(
        system_prompt=_THINKER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    parsed = _parse_thinker_json(raw)

    models = _build_mental_models(parsed.get("mental_models", []))
    heuristics = _ensure_strings(parsed.get("decision_heuristics", []))
    values = parsed.get("values_antipatterns", {})
    if not isinstance(values, dict):
        values = {}
    boundaries = _ensure_strings(parsed.get("honest_boundaries", []))

    return {
        "mental_models": models,
        "decision_heuristics": heuristics,
        "values_antipatterns": values,
        "honest_boundaries": boundaries,
    }


def _parse_thinker_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _build_mental_models(raw: list[dict]) -> list[MentalModel]:
    result: list[MentalModel] = []
    for item in raw[:7]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        summary = str(item.get("summary", "")).strip()
        if not name or not summary:
            continue
        result.append(
            MentalModel(
                name=name,
                summary=summary,
                evidence_snippet=str(item.get("evidence_snippet", "")),
                application=str(item.get("application", "")),
                limitation=str(item.get("limitation", "")),
            )
        )
    return result


def _ensure_strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if isinstance(v, str) and v.strip()]
