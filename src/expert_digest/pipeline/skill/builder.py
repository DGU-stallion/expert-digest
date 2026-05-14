"""SKILL.md assembly from all extracted components (deterministic)."""

from __future__ import annotations

from expert_digest.pipeline.state import DigestState


def run_build_skill_md(state: DigestState) -> dict:
    """Assemble final SKILL.md from all extracted components in state."""
    print("  [skill] build_skill_md: assembling markdown...")
    author = state.get("author", "")
    themes = state.get("themes", [])

    sections: list[str] = []

    # ── Title ──
    theme_labels = [t.label for t in themes[:3]]
    theme_tag = "、".join(theme_labels) if theme_labels else ""
    title = f"# {author} · {theme_tag}" if theme_tag else f"# {author}"
    sections.append(title)

    # ── Role rules + identity card + expression DNA (from expresser) ──
    role_rules = state.get("role_rules", "").strip()
    if role_rules:
        sections.append(role_rules)

    # ── Agentic Protocol (from protocol) ──
    protocol_steps = state.get("protocol_steps", "").strip()
    if protocol_steps:
        sections.append(f"## 回答工作流（Agentic Protocol）\n{protocol_steps}")

    # ── Core mental models ──
    models = state.get("mental_models", [])
    if models:
        model_lines: list[str] = ["## 核心心智模型"]
        for i, m in enumerate(models, 1):
            model_lines.append(f"### 模型{i}: {m.name}")
            model_lines.append(f"- 一句话：{m.summary}")
            if m.evidence_snippet:
                model_lines.append(f"- 证据：{m.evidence_snippet}")
            if m.application:
                model_lines.append(f"- 应用：{m.application}")
            if m.limitation:
                model_lines.append(f"- 局限：{m.limitation}")
            model_lines.append("")
        sections.append("\n".join(model_lines).rstrip())

    # ── Decision heuristics ──
    heuristics = state.get("decision_heuristics", [])
    if heuristics:
        h_lines = ["## 决策启发式"]
        for i, h in enumerate(heuristics, 1):
            h_lines.append(f"{i}. {h}")
        sections.append("\n".join(h_lines))

    # ── Values & antipatterns ──
    values = state.get("values_antipatterns", {})
    if isinstance(values, dict) and any(values.values()):
        v_lines = ["## 价值观与反模式"]
        pursues = values.get("pursues", [])
        if isinstance(pursues, list) and pursues:
            v_lines.append("### 追求")
            for p in pursues:
                v_lines.append(f"- {p}")
        opposes = values.get("opposes", [])
        if isinstance(opposes, list) and opposes:
            v_lines.append("### 反对")
            for o in opposes:
                v_lines.append(f"- {o}")
        tensions = values.get("tensions", [])
        if isinstance(tensions, list) and tensions:
            v_lines.append("### 内在张力")
            for t in tensions:
                v_lines.append(f"- {t}")
        sections.append("\n".join(v_lines))

    # ── Honest boundaries ──
    boundaries = state.get("honest_boundaries", [])
    if boundaries:
        b_lines = ["## 诚实边界"]
        for b in boundaries:
            b_lines.append(f"- {b}")
        sections.append("\n".join(b_lines))

    # ── Recommended reading ──
    documents = state.get("documents", [])
    if documents:
        r_lines = ["## 推荐阅读"]
        for d in documents[:10]:
            title = d.get("title", "").strip()
            url = d.get("url", "").strip()
            if title:
                if url:
                    r_lines.append(f"- [{title}]({url})")
                else:
                    r_lines.append(f"- {title}")
        if len(r_lines) > 1:
            sections.append("\n".join(r_lines))

    skill_md = "\n\n".join(sections)
    return {"skill_markdown": skill_md}
