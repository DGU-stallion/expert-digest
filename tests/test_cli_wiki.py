import json

from expert_digest.cli import main
from expert_digest.ingest.jsonl_loader import load_jsonl_documents
from expert_digest.storage.sqlite_store import save_documents


def test_cli_build_evidence_and_build_wiki(tmp_path, capsys):
    db_path = tmp_path / "expert.sqlite3"
    wiki_root = tmp_path / "wiki" / "huang"
    jsonl_path = tmp_path / "articles.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "author": "黄彦臻",
                "title": "泡泡玛特复盘",
                "content": "泡泡玛特的核心能力是 IP 运营。因为它能持续制造角色资产。",
                "source": "sample",
                "url": "https://example.com/popmart",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    save_documents(db_path, load_jsonl_documents(jsonl_path))

    assert main(["build-evidence", "--db", str(db_path)]) == 0
    assert main(
        [
            "build-wiki",
            "--db",
            str(db_path),
            "--wiki-root",
            str(wiki_root),
            "--expert-id",
            "huang",
            "--expert-name",
            "黄彦臻",
            "--purpose",
            "沉淀公开文章。",
        ]
    ) == 0

    output = capsys.readouterr().out
    assert "Built evidence" in output
    assert "Built wiki" in output
    assert (wiki_root / "sources").is_dir()
    assert list((wiki_root / "sources").glob("*.md"))


def test_cli_search_wiki_outputs_hits(tmp_path, capsys):
    db_path = tmp_path / "expert.sqlite3"
    wiki_root = tmp_path / "wiki" / "huang"
    jsonl_path = tmp_path / "articles.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "author": "黄彦臻",
                "title": "泡泡玛特复盘",
                "content": "泡泡玛特的核心能力是 IP 运营。",
                "source": "sample",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    save_documents(db_path, load_jsonl_documents(jsonl_path))

    main(["build-evidence", "--db", str(db_path)])
    main(
        [
            "build-wiki",
            "--db",
            str(db_path),
            "--wiki-root",
            str(wiki_root),
            "--expert-id",
            "huang",
            "--expert-name",
            "黄彦臻",
            "--purpose",
            "沉淀公开文章。",
        ]
    )
    assert main(["search-wiki", "泡泡玛特", "--wiki-root", str(wiki_root)]) == 0

    output = capsys.readouterr().out
    assert "泡泡玛特" in output


def test_cli_lint_wiki_outputs_report(tmp_path, capsys):
    db_path = tmp_path / "expert.sqlite3"
    wiki_root = tmp_path / "wiki" / "huang"
    jsonl_path = tmp_path / "articles.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "author": "黄彦臻",
                "title": "泡泡玛特复盘",
                "content": "泡泡玛特的核心能力是 IP 运营。",
                "source": "sample",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    save_documents(db_path, load_jsonl_documents(jsonl_path))

    main(["build-evidence", "--db", str(db_path)])
    main(
        [
            "build-wiki",
            "--db",
            str(db_path),
            "--wiki-root",
            str(wiki_root),
            "--expert-id",
            "huang",
            "--expert-name",
            "黄彦臻",
            "--purpose",
            "沉淀公开文章。",
        ]
    )
    assert main(["lint-wiki", "--wiki-root", str(wiki_root)]) == 0

    output = capsys.readouterr().out
    assert "low_info_title_pages" in output
    assert "near_duplicate_title_groups" in output
