"""Main pipeline graph composing analysis, handbook, and SKILL subgraphs."""

from __future__ import annotations

from pathlib import Path

from langgraph.graph import END, StateGraph

from expert_digest.pipeline.handbook.graph import build_handbook_subgraph
from expert_digest.pipeline.nodes.analyzer import run_analyze_content
from expert_digest.pipeline.nodes.clusterer import run_cluster_content
from expert_digest.pipeline.nodes.expression import run_analyze_expression
from expert_digest.pipeline.nodes.loader import run_load_data
from expert_digest.pipeline.nodes.quality import (
    run_assess_quality,
    should_retry_analysis,
)
from expert_digest.pipeline.skill.graph import build_skill_subgraph
from expert_digest.pipeline.state import DigestState, PipelineError


def build_main_graph() -> StateGraph:
    """Build the full pipeline graph.

    Graph structure:
      entry → analyze_content → analyze_expression → assess_quality
        → (retry) → analyze_content (loop)
        → (proceed) → handbook subgraph + skill subgraph (parallel)
    """
    builder = StateGraph(DigestState)

    # ── Analysis nodes ──
    builder.add_node("entry", run_load_data)
    builder.add_node("cluster_content", run_cluster_content)
    builder.add_node("analyze_content", run_analyze_content)
    builder.add_node("analyze_expression", run_analyze_expression)
    builder.add_node("assess_quality", run_assess_quality)

    # ── Subgraphs (compiled) ──
    hb_subgraph = build_handbook_subgraph().compile()
    builder.add_node("handbook_pipeline", hb_subgraph)

    sk_subgraph = build_skill_subgraph().compile()
    builder.add_node("skill_pipeline", sk_subgraph)

    # ── Output nodes ──
    builder.add_node("output_handbook", _output_handbook)
    builder.add_node("output_skill", _output_skill)

    # ── Fan-out router ──
    builder.add_node("route_to_products", _route_to_products)

    # ── Edges ──
    builder.set_entry_point("entry")
    builder.add_edge("entry", "cluster_content")
    builder.add_edge("cluster_content", "analyze_content")
    builder.add_edge("analyze_content", "analyze_expression")
    builder.add_edge("analyze_expression", "assess_quality")
    builder.add_conditional_edges(
        "assess_quality",
        should_retry_analysis,
        {"retry": "analyze_content", "proceed": "route_to_products"},
    )

    # Sequential: handbook → skill (avoid parallel LastValue writes to shared state).
    builder.add_edge("route_to_products", "handbook_pipeline")
    builder.add_edge("handbook_pipeline", "skill_pipeline")
    builder.add_edge("skill_pipeline", "output_handbook")
    builder.add_edge("skill_pipeline", "output_skill")
    builder.add_edge("output_handbook", END)
    builder.add_edge("output_skill", END)

    return builder


def compile_pipeline() -> object:
    """Build and compile the full pipeline graph."""
    graph = build_main_graph()
    return graph.compile()


def _output_handbook(state: DigestState) -> dict:
    """Write handbook markdown to the output directory."""
    handbook_md = state.get("handbook_markdown", "").strip()
    if not handbook_md:
        return {"errors": state.get("errors", [])}

    output_dir = Path(state.get("output_dir", "data/outputs"))
    output_path = output_dir / "handbook.md"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path.write_text(handbook_md, encoding="utf-8")
    except OSError as e:
        return {"errors": state.get("errors", []) + [PipelineError(node="output_handbook", message=str(e))]}
    return {}


def _route_to_products(state: DigestState) -> dict:
    """No-op router that fans out to handbook and skill subgraphs."""
    _ = state
    return {}


def _output_skill(state: DigestState) -> dict:
    """Write skill markdown to the output directory.

    Only writes when _skill_verified is True, so test pipeline invocations
    (which produce stub skill content) do not overwrite production output.
    """
    skill_md = state.get("skill_markdown", "").strip()
    verified = state.get("_skill_verified", False)
    if not skill_md or not verified:
        return {}

    output_dir = Path(state.get("output_dir", "data/outputs"))
    output_path = output_dir / "skill.md"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path.write_text(skill_md, encoding="utf-8")
    except OSError as e:
        return {"errors": state.get("errors", []) + [PipelineError(node="output_skill", message=str(e))]}
    return {}
