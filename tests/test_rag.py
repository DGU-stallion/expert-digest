from expert_digest.rag.answering import build_structured_answer
from expert_digest.retrieval.retriever import RetrievedChunk


def _chunk(
    *,
    chunk_id: str,
    score: float,
    title: str = "默认标题",
    author: str = "默认作者",
    text: str = "默认正文片段",
    url: str | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        score=score,
        document_id=f"doc-{chunk_id}",
        title=title,
        author=author,
        text=text,
        url=url,
    )


def test_build_structured_answer_returns_answer_evidence_and_references():
    result = build_structured_answer(
        question="泡泡玛特的核心能力是什么？",
        evidence_chunks=[
            _chunk(
                chunk_id="c1",
                score=0.88,
                title="泡泡玛特复盘",
                author="黄彦臻",
                text="泡泡玛特的核心在于IP运营与预期管理。",
                url="https://example.com/p1",
            ),
            _chunk(
                chunk_id="c2",
                score=0.65,
                title="品牌策略",
                author="黄彦臻",
                text="品牌长期价值来自稳定供给和新品节奏。",
                url=None,
            ),
        ],
    )

    assert result.refused is False
    assert result.answer
    assert len(result.evidence) == 2
    assert result.evidence[0].chunk_id == "c1"
    assert "泡泡玛特复盘" in result.recommended_original[0]
    assert result.uncertainty


def test_build_structured_answer_refuses_when_no_evidence():
    result = build_structured_answer(
        question="这家公司明年营收会是多少？",
        evidence_chunks=[],
    )

    assert result.refused is True
    assert "无法基于当前知识库回答" in result.answer
    assert result.evidence == []
    assert result.recommended_original == []
    assert "未检索到相关证据" in result.uncertainty
