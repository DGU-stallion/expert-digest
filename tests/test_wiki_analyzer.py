from expert_digest.domain.models import Document
from expert_digest.processing.evidence_builder import build_document_evidence
from expert_digest.wiki.analyzer import analyze_document_evidence


def test_analyze_document_evidence_extracts_claims_concepts_and_topics():
    document = Document.create(
        author="黄彦臻",
        title="泡泡玛特复盘",
        content=(
            "泡泡玛特的核心能力是 IP 运营。"
            "因为它能持续制造角色资产，所以估值不能只看玩具销售。"
        ),
        source="sample",
        url="https://example.com/popmart",
    )
    evidence = build_document_evidence(document, span_max_chars=40)

    analysis = analyze_document_evidence(evidence)

    assert analysis.source_id == document.id
    assert analysis.source_title == "泡泡玛特复盘"
    assert "泡泡玛特" in analysis.concepts
    assert "IP" in analysis.concepts or "运营" in analysis.concepts
    assert analysis.key_claims
    assert analysis.evidence_span_ids
    assert analysis.topics


def test_analyze_document_evidence_marks_low_confidence_when_no_spans():
    document = Document.create(
        author="黄彦臻",
        title="空文章",
        content="",
        source="sample",
    )
    evidence = build_document_evidence(document)

    analysis = analyze_document_evidence(evidence)

    assert analysis.confidence == "low"
    assert analysis.key_claims == []
