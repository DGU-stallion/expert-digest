"""SKILL distillation subgraph (LangGraph StateGraph)."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from expert_digest.pipeline.skill.builder import run_build_skill_md
from expert_digest.pipeline.skill.expresser import run_encode_expression
from expert_digest.pipeline.skill.protocol import run_design_protocol
from expert_digest.pipeline.skill.thinker import run_extract_mental_models
from expert_digest.pipeline.skill.verifier import run_verify_skill, skill_verdict
from expert_digest.pipeline.state import DigestState


def build_skill_subgraph() -> StateGraph:
    """Build the SKILL distillation subgraph.

    Graph structure:
      extract_mental_models → encode_expression → design_protocol
        → build_skill_md → verify_skill → output (END)
    """
    builder = StateGraph(DigestState)

    builder.add_node("extract_mental_models", run_extract_mental_models)
    builder.add_node("encode_expression", run_encode_expression)
    builder.add_node("design_protocol", run_design_protocol)
    builder.add_node("build_skill_md", run_build_skill_md)
    builder.add_node("verify_skill", run_verify_skill)

    builder.set_entry_point("extract_mental_models")
    builder.add_edge("extract_mental_models", "encode_expression")
    builder.add_edge("encode_expression", "design_protocol")
    builder.add_edge("design_protocol", "build_skill_md")
    builder.add_edge("build_skill_md", "verify_skill")
    builder.add_conditional_edges(
        "verify_skill",
        skill_verdict,
        {"output": END},
    )

    return builder
