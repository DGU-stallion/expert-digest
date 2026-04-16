import json
from pathlib import Path

from expert_digest.cli import main
from expert_digest.domain.models import Chunk, ChunkEmbedding, Document, Handbook
from expert_digest.retrieval.retriever import ScoredChunk
from expert_digest.storage.sqlite_store import (
    list_chunk_embeddings,
    list_chunks_for_document,
    list_documents,
    save_documents,
)


def test_cli_import_jsonl_saves_documents(tmp_path, capsys):
    jsonl_path = tmp_path / "articles.jsonl"
    db_path = tmp_path / "expert_digest.sqlite3"
    jsonl_path.write_text(
        json.dumps(
            {
                "author": "赵老师",
                "title": "问题意识",
                "content": "先定义问题，再寻找答案。",
                "source": "sample",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    exit_code = main(["import-jsonl", str(jsonl_path), "--db", str(db_path)])

    assert exit_code == 0
    assert "Imported 1 document(s)" in capsys.readouterr().out
    assert [document.title for document in list_documents(db_path)] == ["问题意识"]


def test_cli_import_markdown_saves_documents(tmp_path, capsys):
    markdown_folder = tmp_path / "markdown"
    markdown_folder.mkdir()
    db_path = tmp_path / "expert_digest.sqlite3"
    (markdown_folder / "article.md").write_text(
        "# 行动反馈\n\n行动之后才有反馈。",
        encoding="utf-8",
    )

    exit_code = main(["import-markdown", str(markdown_folder), "--db", str(db_path)])

    assert exit_code == 0
    assert "Imported 1 document(s)" in capsys.readouterr().out
    assert [document.title for document in list_documents(db_path)] == ["行动反馈"]


def test_cli_list_documents_filters_by_author(tmp_path, capsys):
    db_path = tmp_path / "expert_digest.sqlite3"
    save_documents(
        db_path,
        [
            Document.create(
                author="赵老师",
                title="问题意识",
                content="先定义问题，再寻找答案。",
                source="sample",
            ),
            Document.create(
                author="钱老师",
                title="长期反馈",
                content="长期反馈需要稳定记录。",
                source="sample",
            ),
        ],
    )

    exit_code = main(["list-documents", "--author", "赵老师", "--db", str(db_path)])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "问题意识" in output
    assert "长期反馈" not in output


def test_cli_import_zhihu_saves_documents(tmp_path, capsys):
    export_dir = tmp_path / "zhihu-export"
    index_dir = export_dir / "index"
    index_dir.mkdir(parents=True)
    db_path = tmp_path / "expert_digest.sqlite3"
    (index_dir / "content_index.jsonl").write_text(
        json.dumps(
            {
                "source_type": "article",
                "source_id": "a-1",
                "author_name": "黄彦臻",
                "title": "关于泡泡玛特的极简复盘",
                "url": "https://zhuanlan.zhihu.com/p/2023428236160280283",
                "created_at": "2026-04-09T07:30:19.000Z",
                "content_text": "这是正文",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    exit_code = main(["import-zhihu", str(export_dir), "--db", str(db_path)])

    assert exit_code == 0
    assert "Imported 1 document(s)" in capsys.readouterr().out
    assert [document.title for document in list_documents(db_path)] == [
        "关于泡泡玛特的极简复盘"
    ]


def test_cli_build_chunks_saves_chunks(tmp_path, capsys):
    db_path = tmp_path / "expert_digest.sqlite3"
    document = Document.create(
        author="黄彦臻",
        title="测试文章",
        content="第一段内容。第二段内容。第三段内容。",
        source="sample",
    )
    save_documents(db_path, [document])

    exit_code = main(["build-chunks", "--db", str(db_path), "--max-chars", "8"])

    assert exit_code == 0
    assert "Built" in capsys.readouterr().out
    chunks = list_chunks_for_document(db_path, document.id)
    assert len(chunks) >= 2


def test_cli_rebuild_chunks_replaces_old_chunk_set(tmp_path, capsys):
    db_path = tmp_path / "expert_digest.sqlite3"
    document = Document.create(
        author="黄彦臻",
        title="重建测试",
        content="第一段内容。第二段内容。第三段内容。",
        source="sample",
    )
    save_documents(db_path, [document])

    assert main(["build-chunks", "--db", str(db_path), "--max-chars", "6"]) == 0
    old_count = len(list_chunks_for_document(db_path, document.id))
    assert old_count > 1

    exit_code = main(
        [
            "rebuild-chunks",
            "--db",
            str(db_path),
            "--max-chars",
            "100",
            "--min-chars",
            "10",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Rebuilt" in output
    new_chunks = list_chunks_for_document(db_path, document.id)
    assert len(new_chunks) == 1


def test_cli_build_embeddings_and_search_chunks(tmp_path, capsys):
    db_path = tmp_path / "expert_digest.sqlite3"
    document = Document.create(
        author="黄彦臻",
        title="泡泡玛特复盘",
        content="泡泡玛特的核心在于IP运营与预期管理。",
        source="sample",
    )
    save_documents(db_path, [document])
    assert main(["rebuild-chunks", "--db", str(db_path), "--max-chars", "120"]) == 0

    build_exit = main(["build-embeddings", "--db", str(db_path)])
    output = capsys.readouterr().out

    assert build_exit == 0
    assert "Embedded" in output
    assert len(list_chunk_embeddings(db_path, model="hash-bow-v1")) >= 1

    search_exit = main(
        [
            "search-chunks",
            "IP 运营",
            "--db",
            str(db_path),
            "--top-k",
            "1",
        ]
    )
    search_output = capsys.readouterr().out
    assert search_exit == 0
    assert "score=" in search_output


def test_cli_ask_refuses_when_no_embeddings(monkeypatch, capsys):
    monkeypatch.setattr("expert_digest.cli.list_chunk_embeddings", lambda *_a, **_k: [])

    exit_code = main(["ask", "什么是长期主义？"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "回答:" in output
    assert "无法基于当前知识库回答" in output
    assert "依据:" in output
    assert "推荐原文:" in output
    assert "不确定性:" in output


def test_cli_ask_returns_structured_answer_with_evidence(monkeypatch, capsys):
    document = Document.create(
        author="黄彦臻",
        title="泡泡玛特复盘",
        content="泡泡玛特的核心在于IP运营与预期管理。",
        source="sample",
        url="https://example.com/p1",
    )
    chunk = Chunk.create(
        document_id=document.id,
        text="泡泡玛特的核心在于IP运营与预期管理。",
        chunk_index=0,
    )
    embedding = ChunkEmbedding.create(
        chunk_id=chunk.id,
        model="hash-bow-v1",
        vector=[1.0, 0.0],
    )
    monkeypatch.setattr(
        "expert_digest.cli.list_chunk_embeddings",
        lambda *_a, **_k: [embedding],
    )
    monkeypatch.setattr("expert_digest.cli.list_chunks", lambda *_a, **_k: [chunk])
    monkeypatch.setattr(
        "expert_digest.cli.list_documents",
        lambda *_a, **_k: [document],
    )
    monkeypatch.setattr("expert_digest.cli.embed_text", lambda *_a, **_k: [1.0, 0.0])
    monkeypatch.setattr(
        "expert_digest.cli.rank_chunk_embeddings",
        lambda **_k: [ScoredChunk(chunk_id=chunk.id, score=0.99)],
    )

    exit_code = main(["ask", "泡泡玛特的核心能力是什么？", "--top-k", "1"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "回答:" in output
    assert "依据:" in output
    assert "score=0.9900" in output
    assert "推荐原文:" in output
    assert "泡泡玛特复盘" in output
    assert "不确定性:" in output


def test_cli_ask_refuses_when_retrieval_score_below_threshold(monkeypatch, capsys):
    document = Document.create(
        author="黄彦臻",
        title="泡泡玛特复盘",
        content="泡泡玛特的核心在于IP运营与预期管理。",
        source="sample",
    )
    chunk = Chunk.create(
        document_id=document.id,
        text="泡泡玛特的核心在于IP运营与预期管理。",
        chunk_index=0,
    )
    embedding = ChunkEmbedding.create(
        chunk_id=chunk.id,
        model="hash-bow-v1",
        vector=[1.0, 0.0],
    )
    monkeypatch.setattr(
        "expert_digest.cli.list_chunk_embeddings",
        lambda *_a, **_k: [embedding],
    )
    monkeypatch.setattr("expert_digest.cli.list_chunks", lambda *_a, **_k: [chunk])
    monkeypatch.setattr(
        "expert_digest.cli.list_documents",
        lambda *_a, **_k: [document],
    )
    monkeypatch.setattr("expert_digest.cli.embed_text", lambda *_a, **_k: [1.0, 0.0])
    monkeypatch.setattr(
        "expert_digest.cli.rank_chunk_embeddings",
        lambda **_k: [ScoredChunk(chunk_id=chunk.id, score=0.01)],
    )

    exit_code = main(["ask", "泡泡玛特的核心能力是什么？", "--top-k", "1"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "无法基于当前知识库回答" in output


def test_cli_ask_supports_json_output(monkeypatch, capsys):
    document = Document.create(
        author="黄彦臻",
        title="泡泡玛特复盘",
        content="泡泡玛特的核心在于IP运营与预期管理。",
        source="sample",
        url="https://example.com/p1",
    )
    chunk = Chunk.create(
        document_id=document.id,
        text="泡泡玛特的核心在于IP运营与预期管理。",
        chunk_index=0,
    )
    embedding = ChunkEmbedding.create(
        chunk_id=chunk.id,
        model="hash-bow-v1",
        vector=[1.0, 0.0],
    )
    monkeypatch.setattr(
        "expert_digest.cli.list_chunk_embeddings",
        lambda *_a, **_k: [embedding],
    )
    monkeypatch.setattr("expert_digest.cli.list_chunks", lambda *_a, **_k: [chunk])
    monkeypatch.setattr(
        "expert_digest.cli.list_documents",
        lambda *_a, **_k: [document],
    )
    monkeypatch.setattr("expert_digest.cli.embed_text", lambda *_a, **_k: [1.0, 0.0])
    monkeypatch.setattr(
        "expert_digest.cli.rank_chunk_embeddings",
        lambda **_k: [ScoredChunk(chunk_id=chunk.id, score=0.99)],
    )

    exit_code = main(
        [
            "ask",
            "泡泡玛特的核心能力是什么？",
            "--top-k",
            "1",
            "--format",
            "json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["refused"] is False
    assert payload["evidence"][0]["chunk_id"] == chunk.id
    assert payload["recommended_original"]


def test_cli_ask_refuses_when_average_score_below_threshold(monkeypatch, capsys):
    document_1 = Document.create(
        author="黄彦臻",
        title="泡泡玛特复盘",
        content="泡泡玛特的核心在于IP运营与预期管理。",
        source="sample",
    )
    document_2 = Document.create(
        author="黄彦臻",
        title="渠道策略",
        content="渠道效率会影响增长速度。",
        source="sample",
    )
    chunk_1 = Chunk.create(
        document_id=document_1.id,
        text="泡泡玛特的核心在于IP运营与预期管理。",
        chunk_index=0,
    )
    chunk_2 = Chunk.create(
        document_id=document_2.id,
        text="渠道效率会影响增长速度。",
        chunk_index=0,
    )
    embeddings = [
        ChunkEmbedding.create(
            chunk_id=chunk_1.id,
            model="hash-bow-v1",
            vector=[1.0, 0.0],
        ),
        ChunkEmbedding.create(
            chunk_id=chunk_2.id,
            model="hash-bow-v1",
            vector=[0.0, 1.0],
        ),
    ]
    monkeypatch.setattr(
        "expert_digest.cli.list_chunk_embeddings",
        lambda *_a, **_k: embeddings,
    )
    monkeypatch.setattr(
        "expert_digest.cli.list_chunks",
        lambda *_a, **_k: [chunk_1, chunk_2],
    )
    monkeypatch.setattr(
        "expert_digest.cli.list_documents",
        lambda *_a, **_k: [document_1, document_2],
    )
    monkeypatch.setattr("expert_digest.cli.embed_text", lambda *_a, **_k: [1.0, 0.0])
    monkeypatch.setattr(
        "expert_digest.cli.rank_chunk_embeddings",
        lambda **_k: [
            ScoredChunk(chunk_id=chunk_1.id, score=0.95),
            ScoredChunk(chunk_id=chunk_2.id, score=0.01),
        ],
    )

    exit_code = main(
        [
            "ask",
            "泡泡玛特的核心能力是什么？",
            "--top-k",
            "2",
            "--min-top-score",
            "0.8",
            "--min-avg-score",
            "0.6",
            "--format",
            "json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["refused"] is True


def test_cli_generate_handbook_writes_output(monkeypatch, capsys):
    captured: dict[str, object] = {}
    output_path = Path("data/outputs/test_cli_generate_handbook.md")
    handbook = Handbook(
        author="黄彦臻",
        title="黄彦臻学习手册",
        markdown="# 手册\n\n## 专家内容总览\n",
        source_document_ids=["doc-1", "doc-2"],
    )

    def _fake_build_handbook(**kwargs):
        captured.update(kwargs)
        return handbook

    def _fake_write_handbook(*, handbook: Handbook, output_path: str | Path) -> Path:
        return Path(output_path)

    monkeypatch.setattr("expert_digest.cli.build_handbook", _fake_build_handbook)
    monkeypatch.setattr("expert_digest.cli.write_handbook", _fake_write_handbook)

    exit_code = main(
        [
            "generate-handbook",
            "--author",
            "黄彦臻",
            "--top-k",
            "4",
            "--max-themes",
            "2",
            "--output",
            str(output_path),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert captured["author"] == "黄彦臻"
    assert captured["top_k"] == 4
    assert captured["max_themes"] == 2
    assert "Generated handbook" in output


def test_cli_generate_handbook_returns_error_on_generation_failure(
    monkeypatch, capsys
):
    def _raise_error(**_kwargs):
        raise ValueError("no documents available for handbook generation")

    monkeypatch.setattr("expert_digest.cli.build_handbook", _raise_error)

    exit_code = main(["generate-handbook"])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "Failed to generate handbook" in output


def test_cli_generate_handbook_hybrid_uses_default_llm_client(monkeypatch, capsys):
    captured: dict[str, object] = {}
    fake_llm_client = object()
    handbook = Handbook(
        author="黄彦臻",
        title="黄彦臻学习手册",
        markdown="# 手册\n",
        source_document_ids=["doc-1"],
    )

    def _fake_create_default_llm_client(**kwargs):
        captured.update(kwargs)
        return fake_llm_client

    def _fake_build_handbook(**kwargs):
        synthesizer = kwargs["synthesizer"]
        assert synthesizer._llm_client is fake_llm_client
        return handbook

    monkeypatch.setattr(
        "expert_digest.cli.create_default_handbook_llm_client",
        _fake_create_default_llm_client,
    )
    monkeypatch.setattr("expert_digest.cli.build_handbook", _fake_build_handbook)
    monkeypatch.setattr(
        "expert_digest.cli.write_handbook",
        lambda *, handbook, output_path: Path(output_path),
    )

    exit_code = main(
        [
            "generate-handbook",
            "--synthesis-mode",
            "hybrid",
            "--ccswitch-db",
            "data/processed/mock_ccswitch.sqlite3",
            "--llm-timeout",
            "12",
            "--llm-max-tokens",
            "600",
        ]
    )
    _ = capsys.readouterr().out

    assert exit_code == 0
    assert captured["ccswitch_db_path"] == Path("data/processed/mock_ccswitch.sqlite3")
    assert captured["timeout_seconds"] == 12
    assert captured["max_output_tokens"] == 600
