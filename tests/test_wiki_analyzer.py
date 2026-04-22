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


def test_analyze_document_evidence_filters_question_template_noise():
    document = Document.create(
        author="黄彦臻",
        title="10 月 9 日 A 股大幅回调，如何看待今日行情？",
        content=(
            "全市场逾 200 只个股下跌。"
            "发生了什么？"
            "请问后续怎么走？"
        ),
        source="sample",
    )
    evidence = build_document_evidence(document, span_max_chars=30)

    analysis = analyze_document_evidence(evidence)

    joined_concepts = " ".join(analysis.concepts)
    joined_topics = " ".join(analysis.topics)
    assert "如何看待今日行情" not in joined_concepts
    assert "发生了什么" not in joined_concepts
    assert "请问后续怎么走" not in joined_concepts
    assert "如何看待今日行情" not in joined_topics
    assert "发生了什么" not in joined_topics
    assert "请问后续怎么走" not in joined_topics


def test_analyze_document_evidence_filters_date_fragment_noise():
    document = Document.create(
        author="黄彦臻",
        title="12 月 4 日午间，比特币价格大跳水，发生了什么？",
        content="比特币价格在一小时内快速回撤。",
        source="sample",
    )
    evidence = build_document_evidence(document, span_max_chars=30)

    analysis = analyze_document_evidence(evidence)

    assert "日午间" not in analysis.concepts
    assert "发生了什么" not in analysis.concepts


def test_analyze_document_evidence_filters_market_broadcast_noise():
    document = Document.create(
        author="黄彦臻",
        title="10 月 9 日午间全市场逾 500 只个股涨停，发生了什么？",
        content=(
            "午间盘中波动放大，全市场逾 500 只个股涨停。"
            "请问后续会怎么走？"
        ),
        source="sample",
    )
    evidence = build_document_evidence(document, span_max_chars=30)

    analysis = analyze_document_evidence(evidence)

    joined_concepts = " ".join(analysis.concepts)
    joined_topics = " ".join(analysis.topics)
    assert "全市场逾" not in joined_concepts
    assert "只个股涨停" not in joined_concepts
    assert "日午间" not in joined_concepts
    assert "全市场逾" not in joined_topics
    assert "只个股涨停" not in joined_topics


def test_analyze_document_evidence_applies_stricter_output_limits():
    document = Document.create(
        author="黄彦臻",
        title="泡泡玛特海外扩张复盘",
        content=(
            "泡泡玛特的核心能力是 IP 运营。"
            "潮玩行业的增长依赖品牌势能、渠道效率、组织协同和内容供给。"
            "公司在海外市场通过门店、联名和社群运营提升复购。"
        ),
        source="sample",
    )
    evidence = build_document_evidence(document, span_max_chars=40)

    analysis = analyze_document_evidence(evidence)

    assert len(analysis.concepts) <= 8
    assert len(analysis.topics) <= 3
    assert "泡泡玛特" in analysis.concepts


def test_analyze_document_evidence_rejects_two_letter_lowercase_noise():
    document = Document.create(
        author="黄彦臻",
        title="BD BP 交易复盘",
        content="BD BP 是论坛里的噪声缩写，不应成为稳定概念。",
        source="sample",
    )
    evidence = build_document_evidence(document, span_max_chars=30)

    analysis = analyze_document_evidence(evidence)

    assert "bd" not in [item.lower() for item in analysis.concepts]
    assert "bp" not in [item.lower() for item in analysis.concepts]
