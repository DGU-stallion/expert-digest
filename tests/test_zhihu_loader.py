import json

import pytest

from expert_digest.ingest.zhihu_loader import load_zhihu_documents


def test_load_zhihu_documents_from_export_directory(tmp_path):
    export_dir = tmp_path / "huang-wei-yan-30"
    index_dir = export_dir / "index"
    index_dir.mkdir(parents=True)
    path = index_dir / "content_index.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "source_type": "article",
                        "source_id": "100",
                        "author_name": "黄彦臻",
                        "title": "关于泡泡玛特的极简复盘",
                        "url": "https://zhuanlan.zhihu.com/p/100",
                        "created_at": "2026-04-09T07:30:19.000Z",
                        "content_text": "这是正文",
                    },
                    ensure_ascii=False,
                ),
                "",
                json.dumps(
                    {
                        "source_type": "answer",
                        "source_id": "200",
                        "author_name": "黄彦臻",
                        "title": "怎么看待A股行情？",
                        "url": "https://www.zhihu.com/question/1/answer/200",
                        "created_at": "2026-04-08T05:01:38.000Z",
                        "content_markdown": "回答正文",
                    },
                    ensure_ascii=False,
                ),
            ]
        ),
        encoding="utf-8",
    )

    documents = load_zhihu_documents(export_dir)

    assert len(documents) == 2
    assert documents[0].author == "黄彦臻"
    assert documents[0].title == "关于泡泡玛特的极简复盘"
    assert documents[0].content == "这是正文"
    assert documents[0].source == "zhihu:article:100"
    assert documents[1].source == "zhihu:answer:200"
    assert documents[1].content == "回答正文"


def test_load_zhihu_documents_raises_for_missing_required_field(tmp_path):
    path = tmp_path / "content_index.jsonl"
    path.write_text(
        json.dumps(
            {
                "source_type": "article",
                "source_id": "100",
                "author_name": "黄彦臻",
                "content_text": "没有标题",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="line 1.*title"):
        load_zhihu_documents(path)
