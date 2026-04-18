from pathlib import Path

from expert_digest.app.services import SkillDraftResult
from expert_digest.domain.models import Chunk, ChunkEmbedding, Document
from expert_digest.knowledge.topic_clusterer import TopicCluster
from expert_digest.mcp.toolkit import MCPToolkit


def test_mcp_toolkit_ask_author_filters_to_selected_author(monkeypatch):
    doc_a = Document.create(author="作者A", title="A", content="A", source="sample")
    doc_b = Document.create(author="作者B", title="B", content="B", source="sample")
    chunk_a = Chunk.create(document_id=doc_a.id, text="方法论与反馈", chunk_index=0)
    chunk_b = Chunk.create(document_id=doc_b.id, text="其他内容", chunk_index=0)
    emb_a = ChunkEmbedding.create(
        chunk_id=chunk_a.id,
        model="hash-bow-v1",
        vector=[1, 0],
    )
    emb_b = ChunkEmbedding.create(
        chunk_id=chunk_b.id,
        model="hash-bow-v1",
        vector=[0, 1],
    )

    monkeypatch.setattr(
        "expert_digest.mcp.toolkit.get_documents_by_author",
        lambda *_a, **_k: [doc_a],
    )
    monkeypatch.setattr(
        "expert_digest.mcp.toolkit.list_chunks",
        lambda *_a, **_k: [chunk_a, chunk_b],
    )
    monkeypatch.setattr(
        "expert_digest.mcp.toolkit.list_chunk_embeddings",
        lambda *_a, **_k: [emb_a, emb_b],
    )
    monkeypatch.setattr(
        "expert_digest.mcp.toolkit.embed_text",
        lambda *_a, **_k: [1, 0],
    )

    toolkit = MCPToolkit(db_path=Path("data/processed/mock.sqlite3"))
    payload = toolkit.ask_author(
        question="核心方法是什么？",
        author_id="作者A",
        top_k=2,
    )

    assert payload["refused"] is False
    assert payload["evidence"]
    assert all(item["author"] == "作者A" for item in payload["evidence"])


def test_mcp_toolkit_search_posts_deduplicates_document_hits(monkeypatch):
    doc_a = Document.create(author="作者A", title="A", content="A", source="sample")
    doc_b = Document.create(author="作者B", title="B", content="B", source="sample")
    chunk_a1 = Chunk.create(document_id=doc_a.id, text="A1", chunk_index=0)
    chunk_a2 = Chunk.create(document_id=doc_a.id, text="A2", chunk_index=1)
    chunk_b = Chunk.create(document_id=doc_b.id, text="B1", chunk_index=0)
    embeddings = [
        ChunkEmbedding.create(
            chunk_id=chunk_a1.id,
            model="hash-bow-v1",
            vector=[1, 0],
        ),
        ChunkEmbedding.create(
            chunk_id=chunk_a2.id,
            model="hash-bow-v1",
            vector=[0.95, 0.05],
        ),
        ChunkEmbedding.create(
            chunk_id=chunk_b.id,
            model="hash-bow-v1",
            vector=[0.7, 0.3],
        ),
    ]

    monkeypatch.setattr(
        "expert_digest.mcp.toolkit.list_documents",
        lambda *_a, **_k: [doc_a, doc_b],
    )
    monkeypatch.setattr(
        "expert_digest.mcp.toolkit.list_chunks",
        lambda *_a, **_k: [chunk_a1, chunk_a2, chunk_b],
    )
    monkeypatch.setattr(
        "expert_digest.mcp.toolkit.list_chunk_embeddings",
        lambda *_a, **_k: embeddings,
    )
    monkeypatch.setattr(
        "expert_digest.mcp.toolkit.embed_text",
        lambda *_a, **_k: [1, 0],
    )

    toolkit = MCPToolkit(db_path=Path("data/processed/mock.sqlite3"))
    payload = toolkit.search_posts(query="反馈", top_k=3)

    assert len(payload["hits"]) == 2
    assert {item["document_id"] for item in payload["hits"]} == {doc_a.id, doc_b.id}


def test_mcp_toolkit_list_topics_uses_author_scope(monkeypatch):
    doc_a = Document.create(author="作者A", title="A", content="A", source="sample")
    doc_b = Document.create(author="作者B", title="B", content="B", source="sample")
    chunk_a = Chunk.create(document_id=doc_a.id, text="A1", chunk_index=0)
    chunk_b = Chunk.create(document_id=doc_b.id, text="B1", chunk_index=0)
    emb_a = ChunkEmbedding.create(
        chunk_id=chunk_a.id,
        model="hash-bow-v1",
        vector=[1, 0],
    )
    emb_b = ChunkEmbedding.create(
        chunk_id=chunk_b.id,
        model="hash-bow-v1",
        vector=[0, 1],
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "expert_digest.mcp.toolkit.get_documents_by_author",
        lambda *_a, **_k: [doc_a],
    )
    monkeypatch.setattr(
        "expert_digest.mcp.toolkit.list_chunks",
        lambda *_a, **_k: [chunk_a, chunk_b],
    )
    monkeypatch.setattr(
        "expert_digest.mcp.toolkit.list_chunk_embeddings",
        lambda *_a, **_k: [emb_a, emb_b],
    )

    def _fake_cluster_chunks_by_embeddings(**kwargs):
        captured["documents_by_id"] = kwargs["documents_by_id"]
        return [
            TopicCluster(
                topic_id="topic-1",
                label="主题1：A",
                chunk_count=1,
                representative_chunk_ids=[chunk_a.id],
                representative_documents=[],
            )
        ]

    monkeypatch.setattr(
        "expert_digest.mcp.toolkit.cluster_chunks_by_embeddings",
        _fake_cluster_chunks_by_embeddings,
    )

    toolkit = MCPToolkit(db_path=Path("data/processed/mock.sqlite3"))
    payload = toolkit.list_topics(author_id="作者A", num_topics=2, top_docs=1)

    assert len(payload["topics"]) == 1
    assert set(captured["documents_by_id"].keys()) == {doc_a.id}


def test_mcp_toolkit_generate_skill_returns_preview(monkeypatch):
    fake_result = SkillDraftResult(
        profile={"author": "作者A", "document_count": 2},
        markdown="# SKILL\n\n## 规则\n- A\n",
        output_path=Path("data/outputs/a_skill.md"),
    )

    monkeypatch.setattr(
        "expert_digest.mcp.toolkit.services.generate_skill_draft",
        lambda **_kwargs: fake_result,
    )

    toolkit = MCPToolkit(db_path=Path("data/processed/mock.sqlite3"))
    payload = toolkit.generate_skill(author_id="作者A")

    assert payload["author"] == "作者A"
    assert payload["document_count"] == 2
    assert payload["output_path"] == "data/outputs/a_skill.md"
    assert "# SKILL" in payload["markdown_preview"]
