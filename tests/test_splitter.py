from expert_digest.domain.models import Document
from expert_digest.processing.splitter import split_document, split_documents


def test_split_document_returns_one_chunk_for_short_text():
    document = Document.create(
        author="黄彦臻",
        title="短文",
        content="这是一个短文本。",
        source="sample",
    )

    chunks = split_document(document, max_chars=100)

    assert len(chunks) == 1
    assert chunks[0].document_id == document.id
    assert chunks[0].chunk_index == 0
    assert chunks[0].text == "这是一个短文本。"


def test_split_document_splits_long_text_by_max_chars():
    document = Document.create(
        author="黄彦臻",
        title="长文",
        content="第一段内容。" * 6 + "\n\n" + "第二段内容。" * 6,
        source="sample",
    )

    chunks = split_document(document, max_chars=30)

    assert len(chunks) >= 2
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert all(len(chunk.text) <= 30 for chunk in chunks)


def test_split_documents_combines_results():
    first = Document.create(
        author="黄彦臻",
        title="A",
        content="A" * 10,
        source="sample",
    )
    second = Document.create(
        author="黄彦臻",
        title="B",
        content="B" * 10,
        source="sample",
    )

    chunks = split_documents([first, second], max_chars=8)

    document_ids = {chunk.document_id for chunk in chunks}
    assert document_ids == {first.id, second.id}


def test_split_document_merges_short_tail_chunks_when_min_chars_is_set():
    document = Document.create(
        author="黄彦臻",
        title="最小块过滤",
        content="A" * 25 + "\n\n" + "B" * 3,
        source="sample",
    )

    chunks = split_document(document, max_chars=12, min_chars=6)

    assert len(chunks) >= 2
    assert all(len(chunk.text) >= 6 for chunk in chunks)
