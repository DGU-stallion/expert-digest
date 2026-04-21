from expert_digest.wiki.models import SourceRef, WikiPage
from expert_digest.wiki.retriever import search_wiki
from expert_digest.wiki.vault import WikiVault


def test_search_wiki_ranks_title_body_and_sources(tmp_path):
    vault = WikiVault(root=tmp_path / "wiki" / "huang")
    vault.initialize(expert_id="huang", expert_name="黄彦臻", purpose="沉淀公开文章。")
    vault.write_page(
        WikiPage(
            path="topics/ip-operation.md",
            page_type="topic",
            title="IP 运营",
            body="泡泡玛特的核心能力是 IP 运营和角色资产。",
            sources=[SourceRef(source_id="doc-1", title="泡泡玛特复盘", evidence_span_ids=["span-1"])],
        )
    )
    vault.write_page(
        WikiPage(
            path="concepts/macro.md",
            page_type="concept",
            title="宏观经济",
            body="这里讨论利率和汇率。",
        )
    )

    hits = search_wiki(vault=vault, query="泡泡玛特核心能力", top_k=3)

    assert hits[0].page.title == "IP 运营"
    assert hits[0].source_ids == ["doc-1"]
    assert hits[0].score > 0


def test_search_wiki_returns_empty_for_no_match(tmp_path):
    vault = WikiVault(root=tmp_path / "wiki" / "huang")
    vault.initialize(expert_id="huang", expert_name="黄彦臻", purpose="沉淀公开文章。")

    assert search_wiki(vault=vault, query="不存在的问题") == []
