"""Handbook generation subgraph (LangGraph StateGraph)."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from expert_digest.pipeline.handbook.editor import run_coherence_pass
from expert_digest.pipeline.handbook.planner import run_plan_chapters
from expert_digest.pipeline.handbook.reviewer import (
    all_chapters_pass,
    run_review_chapters,
)
from expert_digest.pipeline.handbook.tracer import run_build_trace
from expert_digest.pipeline.handbook.writer import run_draft_chapters
from expert_digest.pipeline.state import DigestState


def build_handbook_subgraph() -> StateGraph:
    """Build the handbook generation subgraph.

    Graph structure:
      plan_chapters → draft_chapters → review_chapters
        → (rewrite) → draft_chapters (loop, with feedback)
        → (proceed) → coherence_pass → build_trace → END

    The rewrite loop feeds review issues back to the writer for
    self-correcting chapter generation.
    """
    builder = StateGraph(DigestState)

    builder.add_node("plan_chapters", run_plan_chapters)
    builder.add_node("draft_chapters", run_draft_chapters)
    builder.add_node("review_chapters", run_review_chapters)
    builder.add_node("coherence_pass", run_coherence_pass)
    builder.add_node("build_trace", run_build_trace)

    builder.set_entry_point("plan_chapters")
    builder.add_edge("plan_chapters", "draft_chapters")
    builder.add_edge("draft_chapters", "review_chapters")
    builder.add_conditional_edges(
        "review_chapters",
        all_chapters_pass,
        {
            "rewrite": "draft_chapters",
            "proceed": "coherence_pass",
        },
    )
    builder.add_edge("coherence_pass", "build_trace")
    builder.add_edge("build_trace", END)

    return builder
