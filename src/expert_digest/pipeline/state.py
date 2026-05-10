"""Shared state types for the LangGraph pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

# ── Analysis phase types ──────────────────────────────────────────────


@dataclass
class Theme:
    """A theme extracted from the author's content."""

    label: str
    summary: str
    source_document_ids: list[str] = field(default_factory=list)
    related_concepts: list[str] = field(default_factory=list)


@dataclass
class ExpressionDNA:
    """The author's expression style and communication patterns."""

    sentence_patterns: list[str] = field(default_factory=list)
    high_frequency_phrases: list[str] = field(default_factory=list)
    certainty_spectrum: list[str] = field(default_factory=list)
    citation_habits: str = ""


# ── Handbook phase types ──────────────────────────────────────────────


@dataclass
class ChapterPlan:
    """One chapter in the handbook plan."""

    title: str
    purpose: str
    target_themes: list[str] = field(default_factory=list)
    estimated_sections: int = 3


@dataclass
class ChapterDraft:
    """A single drafted chapter."""

    title: str
    content: str
    section_count: int = 0


@dataclass
class ReviewResult:
    """Quality review result for a chapter."""

    chapter_title: str
    passed: bool
    issues: list[str] = field(default_factory=list)


# ── SKILL phase types ─────────────────────────────────────────────────


@dataclass
class MentalModel:
    """A core mental model extracted from the author's thinking."""

    name: str
    summary: str
    evidence_snippet: str = ""
    application: str = ""
    limitation: str = ""


# ── Pipeline control ──────────────────────────────────────────────────


@dataclass
class PipelineError:
    """A recoverable pipeline error with context."""

    node: str
    message: str


# ── Main state ────────────────────────────────────────────────────────


class DigestState(TypedDict):
    """Shared state flowing through the LangGraph pipeline."""

    # Input
    db_path: str
    wiki_root: str
    author: str

    # Loaded documents (sampled for LLM analysis)
    documents: list[dict[str, Any]]

    # Analysis phase output
    themes: list[Theme]
    concepts: list[str]
    thinking_patterns: list[str]
    topic_clusters: list  # list[TopicCluster] from knowledge.topic_clusterer
    expression_dna: ExpressionDNA | None
    intellectual_genealogy: str
    key_decisions: list[dict]

    # Handbook output
    chapter_plan: list[ChapterPlan]
    chapters: list[ChapterDraft]
    review_results: list[ReviewResult]
    handbook_markdown: str

    # SKILL output
    mental_models: list[MentalModel]
    decision_heuristics: list[str]
    values_antipatterns: dict
    honest_boundaries: list[str]
    role_rules: str
    protocol_steps: str
    skill_markdown: str

    # Verification flags
    _skill_verified: bool

    # Control
    max_rounds: int
    current_round: int
    errors: list[PipelineError]
    output_dir: str


def make_initial_state(
    *,
    db_path: str | Path = "",
    wiki_root: str | Path = "",
    author: str = "",
    max_rounds: int = 3,
    output_dir: str | Path = "data/outputs",
) -> DigestState:
    """Create a DigestState with all defaults for clean pipeline start."""
    return DigestState(
        db_path=str(db_path),
        wiki_root=str(wiki_root),
        author=author,
        documents=[],
        themes=[],
        concepts=[],
        thinking_patterns=[],
        topic_clusters=[],
        expression_dna=None,
        intellectual_genealogy="",
        key_decisions=[],
        chapter_plan=[],
        chapters=[],
        review_results=[],
        handbook_markdown="",
        mental_models=[],
        decision_heuristics=[],
        values_antipatterns={},
        honest_boundaries=[],
        role_rules="",
        protocol_steps="",
        skill_markdown="",
        _skill_verified=False,
        max_rounds=max_rounds,
        current_round=0,
        errors=[],
        output_dir=str(output_dir),
    )
