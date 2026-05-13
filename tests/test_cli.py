import json
from io import StringIO

from expert_digest.cli import _print_json_safely, main
from expert_digest.domain.models import Document
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
    monkeypatch.setattr("expert_digest.cli._load_pipeline_env", lambda: None)

    exit_code = main(["generate-handbook"])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "Failed to generate handbook" in output


def test_cli_generate_handbook_output_empty(monkeypatch, capsys, tmp_path):
    class _FakePipeline:
        def invoke(self, state):
            return {"handbook_markdown": ""}

    monkeypatch.setattr("expert_digest.cli.compile_pipeline", lambda: _FakePipeline())
    monkeypatch.setattr("expert_digest.cli._load_pipeline_env", lambda: None)

    output_path = tmp_path / "handbook.md"
    exit_code = main(
        ["generate-handbook", "--output", str(output_path)]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "handbook output is empty" in output


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
