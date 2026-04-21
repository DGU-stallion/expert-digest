from expert_digest.domain.models import Document
from expert_digest.processing.evidence_builder import build_document_evidence
from expert_digest.wiki.analyzer import analyze_document_evidence
from expert_digest.wiki.vault import WikiVault
from expert_digest.wiki.writer import write_analysis_to_vault


def test_write_analysis_to_vault_creates_source_concept_topic_index_and_log(tmp_path):
    document = Document.create(
        author="黄彦臻",
        title="泡泡玛特复盘",
        content="泡泡玛特的核心能力是 IP 运营。因为它能持续制造角色资产。",
        source="sample",
        url="https://example.com/popmart",
    )
    evidence = build_document_evidence(document, span_max_chars=40)
    analysis = analyze_document_evidence(evidence)
    vault = WikiVault(root=tmp_path / "wiki" / "huang")
    vault.initialize(
        expert_id="huang",
        expert_name="黄彦臻",
        purpose="沉淀公开文章。",
    )

    written = write_analysis_to_vault(
        vault=vault,
        analysis=analysis,
        evidence_spans=evidence.evidence_spans,
    )

    assert "sources" in written
    assert "concepts" in written
    assert "topics" in written
    assert (vault.root / "sources" / f"{document.id}.md").exists()
    assert "泡泡玛特复盘" in (vault.root / "index.md").read_text(encoding="utf-8")
    assert "ingested source" in (vault.root / "log.md").read_text(encoding="utf-8")
