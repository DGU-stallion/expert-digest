"""SKILL quality verification with deterministic checks and refine routing."""

from __future__ import annotations

from expert_digest.pipeline.state import DigestState, PipelineError


def run_verify_skill(state: DigestState) -> dict:
    """Verify SKILL.md quality with three test categories.

    Returns verification results stored in state for routing.
    """
    print("  [skill] verify_skill: quality checking...")
    skill_md = state.get("skill_markdown", "").strip()
    models = state.get("mental_models", [])
    boundaries = state.get("honest_boundaries", [])
    heuristics = state.get("decision_heuristics", [])
    role_rules = state.get("role_rules", "").strip()
    protocol_steps = state.get("protocol_steps", "").strip()
    values = state.get("values_antipatterns", {})

    errors: list[PipelineError] = list(state.get("errors", []))

    if not skill_md:
        errors.append(PipelineError(node="verify_skill", message="skill_markdown is empty"))
        return {"errors": errors, "_skill_verified": False}

    issues: list[str] = []

    # Test 1: Structure check — required sections present
    required = [
        ("心智模型", "核心心智模型"),
        ("角色扮演规则", "角色扮演规则"),
        ("回答工作流", "Agentic Protocol"),
        ("诚实边界", "诚实边界"),
        ("决策启发式", "决策启发式"),
    ]
    for label, keyword in required:
        if keyword not in skill_md:
            issues.append(f"缺少{label}")

    # Test 2: Content depth check
    if len(models) < 3:
        issues.append(f"心智模型数量不足（{len(models)}/3）")
    if len(boundaries) < 3:
        issues.append(f"诚实边界数量不足（{len(boundaries)}/3）")
    if len(heuristics) < 3:
        issues.append(f"决策启发式数量不足（{len(heuristics)}/3）")
    if not role_rules:
        issues.append("缺少角色扮演规则")
    if not protocol_steps:
        issues.append("缺少Agentic Protocol")
    if not isinstance(values, dict) or not any(values.values()):
        issues.append("缺少价值观与反模式")

    # Test 3: Style check — no obvious template artifacts
    if "{{" in skill_md or "}}" in skill_md:
        issues.append("包含未替换的模板变量")
    if len(skill_md) < 500:
        issues.append(f"内容过短（{len(skill_md)}字）")

    if issues:
        errors.append(PipelineError(node="verify_skill", message="; ".join(issues)))
        return {"errors": errors, "_skill_verified": False}

    return {"_skill_verified": True}


def skill_verdict(state: DigestState) -> str:
    """Always route to 'output'. Verification results are recorded as errors."""
    _ = state
    return "output"
