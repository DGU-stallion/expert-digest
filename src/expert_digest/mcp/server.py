"""MCP server entrypoint for Cherry Studio integration."""

from __future__ import annotations

from pathlib import Path

from expert_digest.processing.embedder import DEFAULT_EMBEDDING_MODEL
from expert_digest.storage.sqlite_store import DEFAULT_DATABASE_PATH

from .toolkit import MCPToolkit


def run_mcp_server(
    *,
    db_path: str | Path = DEFAULT_DATABASE_PATH,
    model: str = DEFAULT_EMBEDDING_MODEL,
    output_dir: str | Path = Path("data/outputs"),
    transport: str = "stdio",
) -> None:
    """Start the ExpertDigest MCP server with stdio transport."""
    mcp = _create_fastmcp(toolkit=MCPToolkit(
        db_path=Path(db_path),
        model=model,
        output_dir=Path(output_dir),
    ))
    mcp.run(transport=transport)


def _create_fastmcp(*, toolkit: MCPToolkit):
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as error:  # pragma: no cover
        raise RuntimeError(
            "MCP dependency is missing. Install with `pip install -e \".[mcp]\"`."
        ) from error

    mcp = FastMCP("expert-digest")

    @mcp.tool()
    def ask_author(
        question: str,
        author_id: str | None = None,
        top_k: int = 3,
        max_evidence: int = 3,
    ) -> dict[str, object]:
        """Answer a question with evidence from the local expert knowledge base."""
        return toolkit.ask_author(
            question=question,
            author_id=author_id,
            top_k=top_k,
            max_evidence=max_evidence,
        )

    @mcp.tool()
    def search_posts(
        query: str,
        author_id: str | None = None,
        top_k: int = 5,
    ) -> dict[str, object]:
        """Search relevant source posts and return high-scoring representative hits."""
        return toolkit.search_posts(
            query=query,
            author_id=author_id,
            top_k=top_k,
        )

    @mcp.tool()
    def recommend_readings(
        question: str,
        author_id: str | None = None,
        top_k: int = 3,
    ) -> dict[str, object]:
        """Recommend readings for a question based on retrieved evidence."""
        return toolkit.recommend_readings(
            question=question,
            author_id=author_id,
            top_k=top_k,
        )

    @mcp.tool()
    def list_topics(
        author_id: str | None = None,
        num_topics: int = 3,
        top_docs: int = 3,
    ) -> dict[str, object]:
        """List clustered topics and representative documents."""
        return toolkit.list_topics(
            author_id=author_id,
            num_topics=num_topics,
            top_docs=top_docs,
        )

    @mcp.tool()
    def generate_handbook(
        author_id: str | None = None,
        output_path: str | None = None,
        synthesis_mode: str = "deterministic",
        theme_source: str = "cluster",
        num_topics: int = 3,
    ) -> dict[str, object]:
        """Generate a markdown handbook and return metadata about the artifact."""
        return toolkit.generate_handbook(
            author_id=author_id,
            output_path=output_path,
            synthesis_mode=synthesis_mode,
            theme_source=theme_source,
            num_topics=num_topics,
        )

    @mcp.tool()
    def generate_skill(
        author_id: str | None = None,
        output_path: str | None = None,
    ) -> dict[str, object]:
        """Generate a SKILL.md draft from author profile heuristics."""
        return toolkit.generate_skill(
            author_id=author_id,
            output_path=output_path,
        )

    return mcp
