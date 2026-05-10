import json
from io import StringIO
from pathlib import Path

from expert_digest.cli import _print_json_safely, main
from expert_digest.domain.models import Document, Handbook
from expert_digest.rag.answering import AnswerEvidence, StructuredAnswer
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
    monkeypatch.setattr(
        "expert_digest.cli.answer_question",
        lambda **_kwargs: StructuredAnswer(
            answer="抱歉，我无法基于当前知识库回答这个问题。",
            evidence=[],
            recommended_original=[],
            uncertainty="未检索到相关证据，结论风险过高。",
            refused=True,
        ),
    )

    exit_code = main(["ask", "什么是长期主义？"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "回答:" in output
    assert "无法基于当前知识库回答" in output
    assert "依据:" in output
    assert "推荐原文:" in output
    assert "不确定性:" in output


def test_cli_ask_returns_structured_answer_with_evidence(monkeypatch, capsys):
    monkeypatch.setattr(
        "expert_digest.cli.answer_question",
        lambda **_kwargs: StructuredAnswer(
            answer="针对问题“泡泡玛特的核心能力是什么？”，当前最相关证据指出：泡泡玛特的核心在于IP运营与预期管理。",
            evidence=[
                AnswerEvidence(
                    chunk_id="chunk-1",
                    score=0.99,
                    title="泡泡玛特复盘",
                    author="黄彦臻",
                    snippet="泡泡玛特的核心在于IP运营与预期管理。",
                    url="https://example.com/p1",
                )
            ],
            recommended_original=["泡泡玛特复盘 - https://example.com/p1"],
            uncertainty="仅检索到 1 条证据。",
            refused=False,
        ),
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
    monkeypatch.setattr(
        "expert_digest.cli.answer_question",
        lambda **_kwargs: StructuredAnswer(
            answer="抱歉，我无法基于当前知识库回答这个问题。",
            evidence=[],
            recommended_original=[],
            uncertainty="证据置信度不足。",
            refused=True,
        ),
    )

    exit_code = main(["ask", "泡泡玛特的核心能力是什么？", "--top-k", "1"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "无法基于当前知识库回答" in output


def test_cli_ask_supports_json_output(monkeypatch, capsys):
    monkeypatch.setattr(
        "expert_digest.cli.answer_question",
        lambda **_kwargs: StructuredAnswer(
            answer="可回答",
            evidence=[
                AnswerEvidence(
                    chunk_id="chunk-json",
                    score=0.99,
                    title="泡泡玛特复盘",
                    author="黄彦臻",
                    snippet="泡泡玛特的核心在于IP运营与预期管理。",
                    url="https://example.com/p1",
                )
            ],
            recommended_original=["泡泡玛特复盘 - https://example.com/p1"],
            uncertainty="低",
            refused=False,
        ),
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
    assert payload["evidence"][0]["chunk_id"] == "chunk-json"
    assert payload["recommended_original"]


def test_cli_ask_refuses_when_average_score_below_threshold(monkeypatch, capsys):
    monkeypatch.setattr(
        "expert_digest.cli.answer_question",
        lambda **_kwargs: StructuredAnswer(
            answer="抱歉，我无法基于当前知识库回答这个问题。",
            evidence=[],
            recommended_original=[],
            uncertainty="平均得分不足。",
            refused=True,
        ),
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


def test_cli_generate_handbook_writes_output(monkeypatch, capsys, tmp_path):
    class _FakePipeline:
        def invoke(self, state):
            return {"handbook_markdown": "# Handbook\n\nContent here.\n"}

    monkeypatch.setattr("expert_digest.cli.compile_pipeline", lambda: _FakePipeline())

    output_path = tmp_path / "handbook.md"
    exit_code = main(["generate-handbook", "--output", str(output_path)])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Generated handbook" in output
    assert output_path.exists()


def test_cli_generate_handbook_returns_error_on_pipeline_failure(
    monkeypatch, capsys
):
    class _BrokenPipeline:
        def invoke(self, state):
            raise RuntimeError("pipeline error: no documents")

    monkeypatch.setattr("expert_digest.cli.compile_pipeline", lambda: _BrokenPipeline())

    exit_code = main(["generate-handbook"])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "Failed to generate handbook" in output


def test_cli_generate_handbook_json_output(monkeypatch, capsys, tmp_path):
    class _FakePipeline:
        def invoke(self, state):
            return {"handbook_markdown": "# Handbook\n\nRich content.\n"}

    monkeypatch.setattr("expert_digest.cli.compile_pipeline", lambda: _FakePipeline())

    output_path = tmp_path / "handbook.md"
    exit_code = main(
        [
            "generate-handbook",
            "--format", "json",
            "--output", str(output_path),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["status"] == "generated"
    assert payload["handbook_length"] > 0


def test_cli_generate_handbook_json_output_empty(monkeypatch, capsys, tmp_path):
    class _FakePipeline:
        def invoke(self, state):
            return {"handbook_markdown": ""}

    monkeypatch.setattr("expert_digest.cli.compile_pipeline", lambda: _FakePipeline())

    output_path = tmp_path / "handbook.md"
    exit_code = main(
        [
            "generate-handbook",
            "--format", "json",
            "--output", str(output_path),
            "--author", "TestAuthor",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["status"] == "empty_handbook"


def test_cli_generate_handbook_fails_quality_gate(monkeypatch, capsys):
    monkeypatch.setattr(
        "expert_digest.cli.evaluate_wiki",
        lambda **_kwargs: type(
            "Report",
            (),
            {"traceability_ratio": 1.0, "coverage_ratio": 1.0},
        )(),
    )
    monkeypatch.setattr(
        "expert_digest.cli.lint_wiki",
        lambda **_kwargs: type("Lint", (), {"issue_count": 99})(),
    )
    monkeypatch.setattr(
        "expert_digest.cli.compile_pipeline",
        lambda: (_ for _ in ()).throw(AssertionError("should not build")),
    )

    exit_code = main(
        [
            "generate-handbook",
            "--wiki-root-for-quality",
            "data/wiki/huang_pass1b",
            "--expected-source-count-for-quality",
            "824",
            "--max-lint-issues-for-quality",
            "20",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "Failed quality gate" in output


def test_print_json_safely_falls_back_when_terminal_encoding_rejects_unicode(
    monkeypatch,
):
    def _raise_unicode_error(_text):
        raise UnicodeEncodeError("gbk", "x", 0, 1, "illegal multibyte sequence")

    fallback_stream = StringIO()
    monkeypatch.setattr("builtins.print", _raise_unicode_error)
    monkeypatch.setattr("expert_digest.cli.sys.stdout", fallback_stream)

    _print_json_safely({"text": "\ue1be"})

    assert "\\ue1be" in fallback_stream.getvalue()
