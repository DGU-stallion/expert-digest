"""Analysis quality check node with conditional branching."""

from __future__ import annotations

from expert_digest.pipeline.state import DigestState, PipelineError


def run_assess_quality(state: DigestState) -> dict:
    """Assess analysis quality and signal retry or proceed.

    Checks that the analysis phase produced non-empty results.
    If quality is insufficient, records an error for the retry loop.
    """
    issues: list[str] = []

    themes = state.get("themes", [])
    concepts = state.get("concepts", [])
    patterns = state.get("thinking_patterns", [])

    if len(themes) < 2:
        issues.append(f"insufficient_themes: got {len(themes)}, need >= 2")
    if len(concepts) < 5:
        issues.append(f"insufficient_concepts: got {len(concepts)}, need >= 5")
    if len(patterns) < 2:
        issues.append(f"insufficient_patterns: got {len(patterns)}, need >= 2")

    if issues:
        current_round = state.get("current_round", 0)
        max_rounds = state.get("max_rounds", 3)
        if current_round >= max_rounds:
            issues.append("max_retry_rounds_exceeded")
            return {
                "errors": state.get("errors", [])
                + [
                    PipelineError(node="assess_quality", message="; ".join(issues))
                ],
            }
        # Signal retry by recording current round progress but not clearing issues
        return {
            "current_round": current_round + 1,
            "errors": state.get("errors", [])
            + [
                PipelineError(node="assess_quality", message="; ".join(issues))
            ],
        }

    return {}


def should_retry_analysis(state: DigestState) -> str:
    """Return 'retry' or 'proceed' based on analysis quality."""
    themes = state.get("themes", [])
    concepts = state.get("concepts", [])
    patterns = state.get("thinking_patterns", [])
    current_round = state.get("current_round", 0)
    max_rounds = state.get("max_rounds", 3)

    if current_round >= max_rounds:
        return "proceed"  # give up and proceed with what we have

    if len(themes) >= 2 and len(concepts) >= 5 and len(patterns) >= 2:
        return "proceed"
    return "retry"
