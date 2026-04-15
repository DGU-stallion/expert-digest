import json

import pytest

from expert_digest.ingest.jsonl_loader import load_jsonl_documents


def test_load_jsonl_documents_skips_blank_lines_and_preserves_fields(tmp_path):
    path = tmp_path / "articles.jsonl"
    rows = [
        {
            "author": "李老师",
            "title": "认知杠杆",
            "content": "好的问题会改变行动的方向。",
            "source": "sample",
            "url": "https://example.com/cognition",
            "created_at": "2026-01-02",
        },
        {
            "author": "李老师",
            "title": "复盘",
            "content": "复盘不是找错，而是寻找下一次更好的动作。",
            "source": "sample",
        },
    ]
    path.write_text(
        "\n".join(
            [
                json.dumps(rows[0], ensure_ascii=False),
                "",
                json.dumps(rows[1], ensure_ascii=False),
            ]
        ),
        encoding="utf-8",
    )

    documents = load_jsonl_documents(path)

    assert len(documents) == 2
    assert documents[0].author == "李老师"
    assert documents[0].title == "认知杠杆"
    assert documents[0].url == "https://example.com/cognition"
    assert documents[0].created_at == "2026-01-02"
    assert documents[1].url is None
    assert documents[1].created_at is None


def test_load_jsonl_documents_raises_clear_error_for_missing_required_field(tmp_path):
    path = tmp_path / "articles.jsonl"
    path.write_text(
        json.dumps(
            {
                "author": "李老师",
                "content": "缺少标题。",
                "source": "sample",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="line 1.*title"):
        load_jsonl_documents(path)


def test_load_jsonl_documents_uses_stable_document_ids(tmp_path):
    path = tmp_path / "articles.jsonl"
    row = {
        "author": "李老师",
        "title": "认知杠杆",
        "content": "好的问题会改变行动的方向。",
        "source": "sample",
    }
    path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")

    first = load_jsonl_documents(path)
    second = load_jsonl_documents(path)

    assert first[0].id == second[0].id
