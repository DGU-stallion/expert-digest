from expert_digest.wiki.models import WikiPage
from expert_digest.wiki.vault import WikiVault


def test_initialize_vault_creates_core_files(tmp_path):
    vault = WikiVault(root=tmp_path / "wiki" / "huang")

    vault.initialize(
        expert_id="huang",
        expert_name="黄彦臻",
        purpose="沉淀黄彦臻公开文章中的投资分析框架。",
    )

    assert (vault.root / "purpose.md").exists()
    assert (vault.root / "schema.md").exists()
    assert (vault.root / "index.md").exists()
    assert (vault.root / "log.md").exists()
    assert (vault.root / "sources").is_dir()
    assert (vault.root / "concepts").is_dir()
    assert (vault.root / "topics").is_dir()
    assert (vault.root / "theses").is_dir()
    assert (vault.root / "reviews").is_dir()


def test_write_and_read_page(tmp_path):
    vault = WikiVault(root=tmp_path / "wiki" / "huang")
    vault.initialize(
        expert_id="huang",
        expert_name="黄彦臻",
        purpose="沉淀公开文章。",
    )
    page = WikiPage(
        path="topics/ip-operation.md",
        page_type="topic",
        title="IP 运营",
        body="## 核心判断\n\n泡泡玛特依赖 [[IP运营]]。",
    )

    path = vault.write_page(page)
    loaded = vault.read_page("topics/ip-operation.md")

    assert path == vault.root / "topics" / "ip-operation.md"
    assert loaded.title == "IP 运营"
    assert loaded.body == "## 核心判断\n\n泡泡玛特依赖 [[IP运营]]。"


def test_list_pages_reads_nested_markdown(tmp_path):
    vault = WikiVault(root=tmp_path / "wiki" / "huang")
    vault.initialize(
        expert_id="huang",
        expert_name="黄彦臻",
        purpose="沉淀公开文章。",
    )
    vault.write_page(WikiPage(path="topics/a.md", page_type="topic", title="A", body="A"))
    vault.write_page(WikiPage(path="concepts/b.md", page_type="concept", title="B", body="B"))

    pages = vault.list_pages()
    titles = sorted(page.title for page in pages)

    assert "A" in titles
    assert "B" in titles
