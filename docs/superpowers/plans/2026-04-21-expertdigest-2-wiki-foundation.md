# ExpertDigest 2.0 Wiki Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建 ExpertDigest 2.0 第一阶段的 Wiki Foundation MVP：层级证据模型、Markdown Wiki vault、两步 ingest、基础 Wiki 检索和质量报告。

**Architecture:** 保留当前 Python dataclass + SQLite + CLI 架构，新增 wiki/evidence 模块。第一阶段不绑定向量数据库，不引入 LangGraph，不让 LlamaIndex/LangChain 接管核心逻辑；先以确定性实现打通 `SourceDocument -> ParentSection -> ChildChunk -> EvidenceSpan -> Expert Wiki`。

**Tech Stack:** Python 3.11+, stdlib dataclasses/pathlib/sqlite3/json/re, pytest, ruff, Markdown files with YAML-like frontmatter, existing ExpertDigest CLI.

---

## 范围切分

本计划只实现第一阶段 Wiki Foundation，不实现完整深度问答、向量数据库、LLM Wiki 自动合并和新前端。

本计划完成后，系统应能：

1. 从现有 `Document` 生成 `ParentSection`、`Chunk`、`EvidenceSpan`。
2. 把层级证据保存到 SQLite。
3. 初始化一个专家 Markdown vault。
4. 用确定性 baseline 分析 source 并写入 source/concept/topic/index/log 页面。
5. 搜索 Wiki 页面并回溯来源。
6. 输出 Wiki 质量报告。

## 文件结构

### 新增文件

- `src/expert_digest/processing/evidence_builder.py`
  - 从已清洗 `Document` 生成 parent sections、child chunks、evidence spans。
- `src/expert_digest/wiki/__init__.py`
  - Wiki 包入口。
- `src/expert_digest/wiki/models.py`
  - Wiki page、source ref、analysis result、quality report 等 dataclass。
- `src/expert_digest/wiki/frontmatter.py`
  - 极简 YAML-like frontmatter 读写，不引入 PyYAML。
- `src/expert_digest/wiki/vault.py`
  - 初始化 vault、写页面、读页面、列页面。
- `src/expert_digest/wiki/analyzer.py`
  - 确定性 source analyzer baseline。
- `src/expert_digest/wiki/writer.py`
  - 根据 analysis result 写 source/concept/topic/index/log 页面。
- `src/expert_digest/wiki/retriever.py`
  - Wiki-native retrieval：title/body/source refs 简单打分。
- `src/expert_digest/wiki/evaluator.py`
  - Wiki 质量报告：traceability、coverage、query answerability。
- `tests/test_evidence_models.py`
- `tests/test_evidence_builder.py`
- `tests/test_sqlite_evidence_store.py`
- `tests/test_wiki_frontmatter.py`
- `tests/test_wiki_vault.py`
- `tests/test_wiki_analyzer.py`
- `tests/test_wiki_writer.py`
- `tests/test_wiki_retriever.py`
- `tests/test_wiki_evaluator.py`
- `tests/test_cli_wiki.py`

### 修改文件

- `src/expert_digest/domain/models.py`
  - 新增 `ParentSection`、`EvidenceSpan`，扩展 `Chunk` 支持 `parent_section_id`。
- `src/expert_digest/storage/sqlite_store.py`
  - 新增 parent/evidence 表、保存、读取、清理函数。
- `src/expert_digest/cli.py`
  - 新增 Wiki Foundation CLI 命令。
- `README.md`
  - 新增 Wiki Foundation 使用说明。
- `docs/progress_status.md`
  - 记录 Wiki Foundation 设计进入实施阶段。

---

### Task 1: 扩展领域模型

**Files:**
- Modify: `src/expert_digest/domain/models.py`
- Test: `tests/test_evidence_models.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_evidence_models.py`：

```python
from expert_digest.domain.models import Chunk, Document, EvidenceSpan, ParentSection


def test_parent_section_create_has_stable_id():
    document = Document.create(
        author="黄彦臻",
        title="关于泡泡玛特的极简复盘",
        content="第一段\n\n第二段",
        source="sample",
        url="https://example.com/a",
    )

    first = ParentSection.create(
        document_id=document.id,
        title="关于泡泡玛特的极简复盘",
        text="第一段\n\n第二段",
        section_index=0,
        start_char=0,
        end_char=7,
    )
    second = ParentSection.create(
        document_id=document.id,
        title="关于泡泡玛特的极简复盘",
        text="第一段\n\n第二段",
        section_index=0,
        start_char=0,
        end_char=7,
    )

    assert first.id == second.id
    assert first.document_id == document.id
    assert first.section_index == 0
    assert first.start_char == 0
    assert first.end_char == 7


def test_chunk_can_reference_parent_section():
    chunk = Chunk.create(
        document_id="doc-1",
        text="泡泡玛特的能力是 IP 运营。",
        chunk_index=0,
        parent_section_id="section-1",
        start_char=10,
        end_char=25,
    )

    assert chunk.parent_section_id == "section-1"
    assert chunk.start_char == 10
    assert chunk.end_char == 25


def test_evidence_span_create_has_source_location():
    span = EvidenceSpan.create(
        document_id="doc-1",
        parent_section_id="section-1",
        chunk_id="chunk-1",
        text="核心能力不是单纯卖潮玩。",
        span_index=0,
        start_char=42,
        end_char=54,
    )

    assert span.document_id == "doc-1"
    assert span.parent_section_id == "section-1"
    assert span.chunk_id == "chunk-1"
    assert span.start_char == 42
    assert span.end_char == 54
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_evidence_models.py -q`

Expected: FAIL，错误包含 `cannot import name 'EvidenceSpan'` 或 `unexpected keyword argument 'parent_section_id'`。

- [ ] **Step 3: 实现模型**

修改 `src/expert_digest/domain/models.py`：

```python
@dataclass(frozen=True)
class ParentSection:
    """A larger context section derived from a source document."""

    id: str
    document_id: str
    title: str
    text: str
    section_index: int
    start_char: int | None = None
    end_char: int | None = None

    @classmethod
    def create(
        cls,
        *,
        document_id: str,
        title: str,
        text: str,
        section_index: int,
        start_char: int | None = None,
        end_char: int | None = None,
    ) -> "ParentSection":
        payload = {
            "document_id": document_id,
            "title": title,
            "text": text,
            "section_index": section_index,
            "start_char": start_char,
            "end_char": end_char,
        }
        return cls(id=_stable_hash(payload), **payload)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
```

更新 `Chunk`：

```python
@dataclass(frozen=True)
class Chunk:
    """A text chunk derived from a source document."""

    id: str
    document_id: str
    text: str
    chunk_index: int
    start_char: int | None = None
    end_char: int | None = None
    parent_section_id: str | None = None
```

同时更新 `Chunk.create()` 的签名和 payload：

```python
parent_section_id: str | None = None,
```

```python
"parent_section_id": parent_section_id,
```

新增 `EvidenceSpan`：

```python
@dataclass(frozen=True)
class EvidenceSpan:
    """A minimum source-backed citation span used by wiki claims."""

    id: str
    document_id: str
    parent_section_id: str
    chunk_id: str
    text: str
    span_index: int
    start_char: int | None = None
    end_char: int | None = None

    @classmethod
    def create(
        cls,
        *,
        document_id: str,
        parent_section_id: str,
        chunk_id: str,
        text: str,
        span_index: int,
        start_char: int | None = None,
        end_char: int | None = None,
    ) -> "EvidenceSpan":
        payload = {
            "document_id": document_id,
            "parent_section_id": parent_section_id,
            "chunk_id": chunk_id,
            "text": text,
            "span_index": span_index,
            "start_char": start_char,
            "end_char": end_char,
        }
        return cls(id=_stable_hash(payload), **payload)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_evidence_models.py -q`

Expected: `3 passed`

- [ ] **Step 5: 运行相关旧测试**

Run: `python -m pytest tests/test_models.py tests/test_splitter.py -q`

Expected: PASS。旧的 `Chunk.create()` 调用不传 `parent_section_id` 时仍兼容。

- [ ] **Step 6: 提交**

```powershell
git add src/expert_digest/domain/models.py tests/test_evidence_models.py
git commit -m "feat: add hierarchical evidence models"
```

---

### Task 2: 持久化 ParentSection 和 EvidenceSpan

**Files:**
- Modify: `src/expert_digest/storage/sqlite_store.py`
- Test: `tests/test_sqlite_evidence_store.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_sqlite_evidence_store.py`：

```python
from expert_digest.domain.models import Chunk, Document, EvidenceSpan, ParentSection
from expert_digest.storage.sqlite_store import (
    clear_evidence,
    list_evidence_spans,
    list_parent_sections,
    save_chunks,
    save_documents,
    save_evidence_spans,
    save_parent_sections,
)


def test_save_and_list_parent_sections(tmp_path):
    db_path = tmp_path / "expert.sqlite3"
    document = Document.create(
        author="黄彦臻",
        title="一篇文章",
        content="第一段\n\n第二段",
        source="sample",
    )
    section = ParentSection.create(
        document_id=document.id,
        title="一篇文章",
        text="第一段",
        section_index=0,
        start_char=0,
        end_char=3,
    )

    save_documents(db_path, [document])
    assert save_parent_sections(db_path, [section]) == 1

    sections = list_parent_sections(db_path)
    assert sections == [section]


def test_save_and_list_evidence_spans(tmp_path):
    db_path = tmp_path / "expert.sqlite3"
    document = Document.create(
        author="黄彦臻",
        title="一篇文章",
        content="核心能力是 IP。",
        source="sample",
    )
    section = ParentSection.create(
        document_id=document.id,
        title="一篇文章",
        text=document.content,
        section_index=0,
    )
    chunk = Chunk.create(
        document_id=document.id,
        parent_section_id=section.id,
        text=document.content,
        chunk_index=0,
    )
    span = EvidenceSpan.create(
        document_id=document.id,
        parent_section_id=section.id,
        chunk_id=chunk.id,
        text="核心能力是 IP。",
        span_index=0,
        start_char=0,
        end_char=8,
    )

    save_documents(db_path, [document])
    save_parent_sections(db_path, [section])
    save_chunks(db_path, [chunk])
    assert save_evidence_spans(db_path, [span]) == 1

    spans = list_evidence_spans(db_path)
    assert spans == [span]


def test_clear_evidence_removes_sections_spans_chunks_and_embeddings(tmp_path):
    db_path = tmp_path / "expert.sqlite3"
    document = Document.create(
        author="黄彦臻",
        title="一篇文章",
        content="核心能力是 IP。",
        source="sample",
    )
    section = ParentSection.create(
        document_id=document.id,
        title="一篇文章",
        text=document.content,
        section_index=0,
    )
    chunk = Chunk.create(
        document_id=document.id,
        parent_section_id=section.id,
        text=document.content,
        chunk_index=0,
    )
    span = EvidenceSpan.create(
        document_id=document.id,
        parent_section_id=section.id,
        chunk_id=chunk.id,
        text=document.content,
        span_index=0,
    )

    save_documents(db_path, [document])
    save_parent_sections(db_path, [section])
    save_chunks(db_path, [chunk])
    save_evidence_spans(db_path, [span])

    assert clear_evidence(db_path) == {"parent_sections": 1, "chunks": 1, "evidence_spans": 1}
    assert list_parent_sections(db_path) == []
    assert list_evidence_spans(db_path) == []
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_sqlite_evidence_store.py -q`

Expected: FAIL，错误包含 `cannot import name 'save_parent_sections'`。

- [ ] **Step 3: 更新 SQLite schema**

在 `src/expert_digest/storage/sqlite_store.py` 导入模型：

```python
from expert_digest.domain.models import Chunk, ChunkEmbedding, Document, EvidenceSpan, ParentSection
```

在 `_ensure_schema()` 中新增 `parent_sections`：

```python
connection.execute(
    """
    CREATE TABLE IF NOT EXISTS parent_sections (
        id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL,
        title TEXT NOT NULL,
        text TEXT NOT NULL,
        section_index INTEGER NOT NULL,
        start_char INTEGER,
        end_char INTEGER,
        imported_at TEXT NOT NULL,
        FOREIGN KEY (document_id) REFERENCES documents (id)
    )
    """
)
```

更新 `chunks` 表 schema，添加新列。先保留兼容迁移：

```python
_ensure_column(connection, "chunks", "parent_section_id", "TEXT")
```

新增 `evidence_spans`：

```python
connection.execute(
    """
    CREATE TABLE IF NOT EXISTS evidence_spans (
        id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL,
        parent_section_id TEXT NOT NULL,
        chunk_id TEXT NOT NULL,
        text TEXT NOT NULL,
        span_index INTEGER NOT NULL,
        start_char INTEGER,
        end_char INTEGER,
        imported_at TEXT NOT NULL,
        FOREIGN KEY (document_id) REFERENCES documents (id),
        FOREIGN KEY (parent_section_id) REFERENCES parent_sections (id),
        FOREIGN KEY (chunk_id) REFERENCES chunks (id)
    )
    """
)
```

新增列迁移 helper：

```python
def _ensure_column(
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_type: str,
) -> None:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    existing = {row[1] for row in rows}
    if column_name not in existing:
        connection.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
        )
```

- [ ] **Step 4: 实现 save/list/clear 函数**

新增函数：

```python
def save_parent_sections(
    db_path: str | Path,
    sections: Iterable[ParentSection],
) -> int:
    database_path = Path(db_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    submitted = list(sections)
    with _connect(database_path) as connection:
        _ensure_schema(connection)
        connection.executemany(
            """
            INSERT OR REPLACE INTO parent_sections (
                id, document_id, title, text, section_index,
                start_char, end_char, imported_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            [
                (
                    item.id,
                    item.document_id,
                    item.title,
                    item.text,
                    item.section_index,
                    item.start_char,
                    item.end_char,
                )
                for item in submitted
            ],
        )
    return len(submitted)
```

```python
def list_parent_sections(db_path: str | Path) -> list[ParentSection]:
    database_path = Path(db_path)
    if not database_path.exists():
        return []
    with _connect(database_path) as connection:
        _ensure_schema(connection)
        rows = connection.execute(
            """
            SELECT id, document_id, title, text, section_index, start_char, end_char
            FROM parent_sections
            ORDER BY document_id, section_index, id
            """
        ).fetchall()
    return [_parent_section_from_row(row) for row in rows]
```

```python
def save_evidence_spans(
    db_path: str | Path,
    spans: Iterable[EvidenceSpan],
) -> int:
    database_path = Path(db_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    submitted = list(spans)
    with _connect(database_path) as connection:
        _ensure_schema(connection)
        connection.executemany(
            """
            INSERT OR REPLACE INTO evidence_spans (
                id, document_id, parent_section_id, chunk_id, text,
                span_index, start_char, end_char, imported_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            [
                (
                    item.id,
                    item.document_id,
                    item.parent_section_id,
                    item.chunk_id,
                    item.text,
                    item.span_index,
                    item.start_char,
                    item.end_char,
                )
                for item in submitted
            ],
        )
    return len(submitted)
```

```python
def list_evidence_spans(db_path: str | Path) -> list[EvidenceSpan]:
    database_path = Path(db_path)
    if not database_path.exists():
        return []
    with _connect(database_path) as connection:
        _ensure_schema(connection)
        rows = connection.execute(
            """
            SELECT id, document_id, parent_section_id, chunk_id, text,
                   span_index, start_char, end_char
            FROM evidence_spans
            ORDER BY document_id, parent_section_id, span_index, id
            """
        ).fetchall()
    return [_evidence_span_from_row(row) for row in rows]
```

```python
def clear_evidence(db_path: str | Path) -> dict[str, int]:
    database_path = Path(db_path)
    if not database_path.exists():
        return {"parent_sections": 0, "chunks": 0, "evidence_spans": 0}
    with _connect(database_path) as connection:
        _ensure_schema(connection)
        section_count = connection.execute("SELECT COUNT(*) FROM parent_sections").fetchone()[0]
        chunk_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        span_count = connection.execute("SELECT COUNT(*) FROM evidence_spans").fetchone()[0]
        connection.execute("DELETE FROM evidence_spans")
        connection.execute("DELETE FROM chunk_embeddings")
        connection.execute("DELETE FROM chunks")
        connection.execute("DELETE FROM parent_sections")
    return {
        "parent_sections": section_count,
        "chunks": chunk_count,
        "evidence_spans": span_count,
    }
```

新增 row converter：

```python
def _parent_section_from_row(row) -> ParentSection:
    return ParentSection(
        id=row[0],
        document_id=row[1],
        title=row[2],
        text=row[3],
        section_index=row[4],
        start_char=row[5],
        end_char=row[6],
    )
```

```python
def _evidence_span_from_row(row) -> EvidenceSpan:
    return EvidenceSpan(
        id=row[0],
        document_id=row[1],
        parent_section_id=row[2],
        chunk_id=row[3],
        text=row[4],
        span_index=row[5],
        start_char=row[6],
        end_char=row[7],
    )
```

更新 `save_chunks()` insert 字段和 `_chunk_from_row()`，让 `parent_section_id` 可保存和读取。

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_sqlite_evidence_store.py tests/test_sqlite_store.py -q`

Expected: PASS。

- [ ] **Step 6: 提交**

```powershell
git add src/expert_digest/storage/sqlite_store.py tests/test_sqlite_evidence_store.py
git commit -m "feat: persist hierarchical evidence"
```

---

### Task 3: 构建层级证据

**Files:**
- Create: `src/expert_digest/processing/evidence_builder.py`
- Test: `tests/test_evidence_builder.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_evidence_builder.py`：

```python
from expert_digest.domain.models import Document
from expert_digest.processing.cleaner import clean_document
from expert_digest.processing.evidence_builder import build_document_evidence


def test_build_document_evidence_creates_sections_chunks_and_spans():
    document = Document.create(
        author="黄彦臻",
        title="关于泡泡玛特的极简复盘",
        content="# 核心判断\n\n泡泡玛特的核心能力不是单纯卖潮玩，而是 IP 运营。\n\n第二段继续解释。",
        source="sample",
        url="https://example.com/popmart",
    )

    result = build_document_evidence(
        clean_document(document),
        parent_max_chars=80,
        child_max_chars=40,
        child_min_chars=10,
        span_max_chars=30,
    )

    assert len(result.parent_sections) == 1
    assert result.parent_sections[0].title == "核心判断"
    assert len(result.chunks) >= 2
    assert all(chunk.parent_section_id == result.parent_sections[0].id for chunk in result.chunks)
    assert len(result.evidence_spans) >= 2
    assert all(span.parent_section_id == result.parent_sections[0].id for span in result.evidence_spans)
    assert all(span.chunk_id for span in result.evidence_spans)


def test_build_document_evidence_falls_back_to_document_title_without_heading():
    document = Document.create(
        author="黄彦臻",
        title="没有标题层级的文章",
        content="第一段内容。\n\n第二段内容。",
        source="sample",
    )

    result = build_document_evidence(document, parent_max_chars=100)

    assert result.parent_sections[0].title == "没有标题层级的文章"
    assert result.parent_sections[0].document_id == document.id
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_evidence_builder.py -q`

Expected: FAIL，错误包含 `No module named 'expert_digest.processing.evidence_builder'`。

- [ ] **Step 3: 实现 evidence builder**

创建 `src/expert_digest/processing/evidence_builder.py`：

```python
"""Build hierarchical source evidence for ExpertDigest 2.0 wiki ingest."""

from __future__ import annotations

from dataclasses import dataclass

from expert_digest.domain.models import Chunk, Document, EvidenceSpan, ParentSection


@dataclass(frozen=True)
class DocumentEvidence:
    document: Document
    parent_sections: list[ParentSection]
    chunks: list[Chunk]
    evidence_spans: list[EvidenceSpan]


def build_document_evidence(
    document: Document,
    *,
    parent_max_chars: int = 2400,
    child_max_chars: int = 700,
    child_min_chars: int = 80,
    span_max_chars: int = 220,
) -> DocumentEvidence:
    if parent_max_chars <= 0:
        raise ValueError("parent_max_chars must be > 0")
    if child_max_chars <= 0:
        raise ValueError("child_max_chars must be > 0")
    if child_min_chars <= 0:
        raise ValueError("child_min_chars must be > 0")
    if span_max_chars <= 0:
        raise ValueError("span_max_chars must be > 0")

    parent_sections = _build_parent_sections(
        document,
        parent_max_chars=parent_max_chars,
    )
    chunks: list[Chunk] = []
    spans: list[EvidenceSpan] = []
    chunk_index = 0

    for section in parent_sections:
        child_texts = _split_by_paragraphs(
            section.text,
            max_chars=child_max_chars,
            min_chars=child_min_chars,
        )
        for child_text in child_texts:
            start = document.content.find(child_text)
            end = start + len(child_text) if start >= 0 else None
            chunk = Chunk.create(
                document_id=document.id,
                parent_section_id=section.id,
                text=child_text,
                chunk_index=chunk_index,
                start_char=start if start >= 0 else None,
                end_char=end,
            )
            chunks.append(chunk)
            for span_index, span_text in enumerate(
                _split_sentences(child_text, max_chars=span_max_chars)
            ):
                span_start = document.content.find(span_text)
                span_end = span_start + len(span_text) if span_start >= 0 else None
                spans.append(
                    EvidenceSpan.create(
                        document_id=document.id,
                        parent_section_id=section.id,
                        chunk_id=chunk.id,
                        text=span_text,
                        span_index=span_index,
                        start_char=span_start if span_start >= 0 else None,
                        end_char=span_end,
                    )
                )
            chunk_index += 1

    return DocumentEvidence(
        document=document,
        parent_sections=parent_sections,
        chunks=chunks,
        evidence_spans=spans,
    )


def _build_parent_sections(
    document: Document,
    *,
    parent_max_chars: int,
) -> list[ParentSection]:
    blocks = _split_markdown_heading_blocks(document.content)
    if not blocks:
        blocks = [(document.title, document.content.strip())]

    sections: list[ParentSection] = []
    section_index = 0
    for title, text in blocks:
        for section_text in _split_by_paragraphs(
            text,
            max_chars=parent_max_chars,
            min_chars=1,
        ):
            start = document.content.find(section_text)
            end = start + len(section_text) if start >= 0 else None
            sections.append(
                ParentSection.create(
                    document_id=document.id,
                    title=title or document.title,
                    text=section_text,
                    section_index=section_index,
                    start_char=start if start >= 0 else None,
                    end_char=end,
                )
            )
            section_index += 1
    return sections


def _split_markdown_heading_blocks(text: str) -> list[tuple[str, str]]:
    current_title = ""
    current_lines: list[str] = []
    blocks: list[tuple[str, str]] = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            if current_lines:
                body = "\n".join(current_lines).strip()
                if body:
                    blocks.append((current_title, body))
                current_lines = []
            current_title = stripped.lstrip("#").strip()
            continue
        current_lines.append(line)

    if current_lines:
        body = "\n".join(current_lines).strip()
        if body:
            blocks.append((current_title, body))
    return blocks


def _split_by_paragraphs(text: str, *, max_chars: int, min_chars: int) -> list[str]:
    paragraphs = [item.strip() for item in text.split("\n\n") if item.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = paragraph
    if current:
        chunks.append(current)

    merged: list[str] = []
    for chunk in chunks:
        if merged and len(chunk) < min_chars:
            merged[-1] = f"{merged[-1]}\n\n{chunk}".strip()
        else:
            merged.append(chunk)
    return merged


def _split_sentences(text: str, *, max_chars: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for char in text.replace("\n", " "):
        current += char
        if char in "。！？.!?" or len(current) >= max_chars:
            stripped = current.strip()
            if stripped:
                pieces.append(stripped)
            current = ""
    if current.strip():
        pieces.append(current.strip())
    return pieces
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_evidence_builder.py -q`

Expected: PASS。

- [ ] **Step 5: 运行处理层测试**

Run: `python -m pytest tests/test_cleaner.py tests/test_splitter.py tests/test_evidence_builder.py -q`

Expected: PASS。

- [ ] **Step 6: 提交**

```powershell
git add src/expert_digest/processing/evidence_builder.py tests/test_evidence_builder.py
git commit -m "feat: build hierarchical document evidence"
```

---

### Task 4: 实现 Wiki frontmatter 和页面模型

**Files:**
- Create: `src/expert_digest/wiki/__init__.py`
- Create: `src/expert_digest/wiki/models.py`
- Create: `src/expert_digest/wiki/frontmatter.py`
- Test: `tests/test_wiki_frontmatter.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_wiki_frontmatter.py`：

```python
from expert_digest.wiki.frontmatter import parse_frontmatter, render_frontmatter
from expert_digest.wiki.models import SourceRef, WikiPage


def test_render_and_parse_frontmatter_roundtrip():
    page = WikiPage(
        path="topics/popmart-core.md",
        page_type="topic",
        title="泡泡玛特的核心能力",
        body="## 核心判断\n\n这是正文。",
        sources=[
            SourceRef(
                source_id="doc-1",
                title="原文标题",
                url="https://example.com/a",
                evidence_span_ids=["span-1", "span-2"],
            )
        ],
        confidence="medium",
        updated_at="2026-04-21",
    )

    text = render_frontmatter(page)
    parsed = parse_frontmatter(text)

    assert parsed.page_type == "topic"
    assert parsed.title == "泡泡玛特的核心能力"
    assert parsed.sources[0].source_id == "doc-1"
    assert parsed.sources[0].evidence_span_ids == ["span-1", "span-2"]
    assert parsed.body == "## 核心判断\n\n这是正文。"


def test_parse_page_without_frontmatter_uses_unknown_type():
    parsed = parse_frontmatter("# 普通页面\n\n正文")

    assert parsed.page_type == "unknown"
    assert parsed.title == "普通页面"
    assert parsed.sources == []
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_wiki_frontmatter.py -q`

Expected: FAIL，错误包含 `No module named 'expert_digest.wiki'`。

- [ ] **Step 3: 实现 Wiki 模型**

创建 `src/expert_digest/wiki/__init__.py`：

```python
"""Wiki-first knowledge artifacts for ExpertDigest 2.0."""
```

创建 `src/expert_digest/wiki/models.py`：

```python
"""Wiki domain models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SourceRef:
    source_id: str
    title: str
    url: str | None = None
    evidence_span_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class WikiPage:
    path: str
    page_type: str
    title: str
    body: str
    sources: list[SourceRef] = field(default_factory=list)
    confidence: str = "medium"
    updated_at: str | None = None
```

- [ ] **Step 4: 实现 frontmatter**

创建 `src/expert_digest/wiki/frontmatter.py`：

```python
"""Small YAML-like frontmatter renderer/parser for wiki pages."""

from __future__ import annotations

from expert_digest.wiki.models import SourceRef, WikiPage


def render_frontmatter(page: WikiPage) -> str:
    lines = [
        "---",
        f"type: {page.page_type}",
        f"title: {page.title}",
        f"confidence: {page.confidence}",
    ]
    if page.updated_at:
        lines.append(f"updated_at: {page.updated_at}")
    lines.append("sources:")
    if not page.sources:
        lines.append("  []")
    else:
        for source in page.sources:
            lines.append(f"  - source_id: {source.source_id}")
            lines.append(f"    title: {source.title}")
            if source.url:
                lines.append(f"    url: {source.url}")
            if source.evidence_span_ids:
                joined = ", ".join(source.evidence_span_ids)
                lines.append(f"    evidence_span_ids: [{joined}]")
    lines.append("---")
    lines.append("")
    lines.append(page.body.rstrip())
    return "\n".join(lines).rstrip() + "\n"


def parse_frontmatter(text: str, *, path: str = "") -> WikiPage:
    if not text.startswith("---\n"):
        return _page_without_frontmatter(text=text, path=path)

    marker = text.find("\n---", 4)
    if marker < 0:
        return _page_without_frontmatter(text=text, path=path)

    raw_meta = text[4:marker].strip().splitlines()
    body = text[marker + 4 :].lstrip("\n")
    page_type = "unknown"
    title = ""
    confidence = "medium"
    updated_at: str | None = None
    sources: list[SourceRef] = []
    current: dict[str, object] | None = None

    for raw_line in raw_meta:
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped == "sources:" or stripped == "[]":
            continue
        if stripped.startswith("- source_id:"):
            if current:
                sources.append(_source_from_mapping(current))
            current = {"source_id": stripped.split(":", 1)[1].strip()}
            continue
        if current is not None and line.startswith("    ") and ":" in stripped:
            key, value = stripped.split(":", 1)
            current[key.strip()] = value.strip()
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        if key == "type":
            page_type = value.strip()
        elif key == "title":
            title = value.strip()
        elif key == "confidence":
            confidence = value.strip()
        elif key == "updated_at":
            updated_at = value.strip()

    if current:
        sources.append(_source_from_mapping(current))

    return WikiPage(
        path=path,
        page_type=page_type,
        title=title or _extract_h1(body) or path,
        body=body.rstrip(),
        sources=sources,
        confidence=confidence,
        updated_at=updated_at,
    )


def _source_from_mapping(mapping: dict[str, object]) -> SourceRef:
    raw_span_ids = str(mapping.get("evidence_span_ids", "")).strip()
    evidence_span_ids: list[str] = []
    if raw_span_ids.startswith("[") and raw_span_ids.endswith("]"):
        inner = raw_span_ids[1:-1].strip()
        evidence_span_ids = [
            item.strip() for item in inner.split(",") if item.strip()
        ]
    return SourceRef(
        source_id=str(mapping.get("source_id", "")).strip(),
        title=str(mapping.get("title", "")).strip(),
        url=str(mapping.get("url")).strip() if mapping.get("url") else None,
        evidence_span_ids=evidence_span_ids,
    )


def _page_without_frontmatter(*, text: str, path: str) -> WikiPage:
    return WikiPage(
        path=path,
        page_type="unknown",
        title=_extract_h1(text) or path,
        body=text.rstrip(),
        sources=[],
        confidence="low",
    )


def _extract_h1(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped.removeprefix("# ").strip()
    return None
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_wiki_frontmatter.py -q`

Expected: PASS。

- [ ] **Step 6: 提交**

```powershell
git add src/expert_digest/wiki tests/test_wiki_frontmatter.py
git commit -m "feat: add wiki page frontmatter"
```

---

### Task 5: 实现 Wiki Vault 初始化和读写

**Files:**
- Create: `src/expert_digest/wiki/vault.py`
- Test: `tests/test_wiki_vault.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_wiki_vault.py`：

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_wiki_vault.py -q`

Expected: FAIL，错误包含 `No module named 'expert_digest.wiki.vault'`。

- [ ] **Step 3: 实现 WikiVault**

创建 `src/expert_digest/wiki/vault.py`：

```python
"""Markdown vault filesystem operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from expert_digest.wiki.frontmatter import parse_frontmatter, render_frontmatter
from expert_digest.wiki.models import WikiPage


@dataclass(frozen=True)
class WikiVault:
    root: Path

    def initialize(
        self,
        *,
        expert_id: str,
        expert_name: str,
        purpose: str,
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for folder in ("sources", "concepts", "topics", "theses", "reviews"):
            (self.root / folder).mkdir(parents=True, exist_ok=True)

        self._write_if_missing(
            "purpose.md",
            f"# Purpose\n\n专家：{expert_name}\n\n{purpose}\n",
        )
        self._write_if_missing(
            "schema.md",
            "# Schema\n\n页面类型：source、concept、topic、thesis、review。\n\n核心判断必须包含 source refs。\n",
        )
        self._write_if_missing(
            "index.md",
            f"# {expert_name} Expert Wiki\n\n- Expert ID: `{expert_id}`\n\n## Sources\n\n## Topics\n\n## Concepts\n",
        )
        self._write_if_missing(
            "log.md",
            "# Log\n\n",
        )

    def write_page(self, page: WikiPage) -> Path:
        path = self.root / page.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_frontmatter(page), encoding="utf-8")
        return path

    def read_page(self, relative_path: str | Path) -> WikiPage:
        path = self.root / relative_path
        text = path.read_text(encoding="utf-8")
        return parse_frontmatter(
            text,
            path=Path(relative_path).as_posix(),
        )

    def list_pages(self) -> list[WikiPage]:
        pages: list[WikiPage] = []
        for path in sorted(self.root.rglob("*.md")):
            relative = path.relative_to(self.root).as_posix()
            pages.append(self.read_page(relative))
        return pages

    def append_log(self, line: str) -> None:
        path = self.root / "log.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(line.rstrip() + "\n")

    def _write_if_missing(self, relative_path: str, content: str) -> None:
        path = self.root / relative_path
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_wiki_vault.py tests/test_wiki_frontmatter.py -q`

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add src/expert_digest/wiki/vault.py tests/test_wiki_vault.py
git commit -m "feat: add markdown wiki vault"
```

---

### Task 6: 确定性 Analyze Source baseline

**Files:**
- Create: `src/expert_digest/wiki/analyzer.py`
- Test: `tests/test_wiki_analyzer.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_wiki_analyzer.py`：

```python
from expert_digest.domain.models import Document
from expert_digest.processing.evidence_builder import build_document_evidence
from expert_digest.wiki.analyzer import analyze_document_evidence


def test_analyze_document_evidence_extracts_claims_concepts_and_topics():
    document = Document.create(
        author="黄彦臻",
        title="泡泡玛特复盘",
        content="泡泡玛特的核心能力是 IP 运营。因为它能持续制造角色资产，所以估值不能只看玩具销售。",
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_wiki_analyzer.py -q`

Expected: FAIL，错误包含 `No module named 'expert_digest.wiki.analyzer'`。

- [ ] **Step 3: 扩展 wiki models**

在 `src/expert_digest/wiki/models.py` 添加：

```python
@dataclass(frozen=True)
class SourceAnalysis:
    source_id: str
    source_title: str
    author: str
    url: str | None
    summary: str
    key_claims: list[str]
    concepts: list[str]
    topics: list[str]
    evidence_span_ids: list[str]
    confidence: str
```

- [ ] **Step 4: 实现 analyzer**

创建 `src/expert_digest/wiki/analyzer.py`：

```python
"""Deterministic source analysis baseline for wiki ingest."""

from __future__ import annotations

import re
from collections import Counter

from expert_digest.processing.evidence_builder import DocumentEvidence
from expert_digest.wiki.models import SourceAnalysis

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]{1,}|[\u4e00-\u9fff]{2,}")
_STOPWORDS = {
    "这个",
    "那个",
    "因为",
    "所以",
    "但是",
    "如果",
    "那么",
    "可以",
    "不是",
    "而是",
    "他们",
    "我们",
}


def analyze_document_evidence(evidence: DocumentEvidence) -> SourceAnalysis:
    spans = evidence.evidence_spans
    claims = [_normalize(span.text) for span in spans[:5] if _normalize(span.text)]
    concepts = _extract_concepts(evidence.document.title + "\n" + evidence.document.content)
    topics = _extract_topics(
        title=evidence.document.title,
        concepts=concepts,
    )
    summary = _build_summary(
        title=evidence.document.title,
        claims=claims,
    )
    return SourceAnalysis(
        source_id=evidence.document.id,
        source_title=evidence.document.title,
        author=evidence.document.author,
        url=evidence.document.url,
        summary=summary,
        key_claims=claims,
        concepts=concepts,
        topics=topics,
        evidence_span_ids=[span.id for span in spans[:8]],
        confidence="medium" if claims else "low",
    )


def _extract_concepts(text: str, *, limit: int = 12) -> list[str]:
    counts = Counter()
    for token in _TOKEN_RE.findall(text):
        normalized = token.strip()
        if len(normalized) < 2:
            continue
        if normalized in _STOPWORDS:
            continue
        counts[normalized] += 1
    return [token for token, _ in counts.most_common(limit)]


def _extract_topics(*, title: str, concepts: list[str], limit: int = 4) -> list[str]:
    candidates = []
    for token in re.split(r"[\s:：,，。;；、\-_/|]+", title):
        stripped = token.strip()
        if len(stripped) >= 2 and stripped not in _STOPWORDS:
            candidates.append(stripped)
    candidates.extend(concepts[:limit])
    return list(dict.fromkeys(candidates))[:limit] or ["未分类主题"]


def _build_summary(*, title: str, claims: list[str]) -> str:
    if not claims:
        return f"《{title}》暂无可稳定抽取的核心判断。"
    return f"《{title}》的核心线索：{claims[0]}"


def _normalize(text: str) -> str:
    return " ".join(text.split())
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_wiki_analyzer.py -q`

Expected: PASS。

- [ ] **Step 6: 提交**

```powershell
git add src/expert_digest/wiki/models.py src/expert_digest/wiki/analyzer.py tests/test_wiki_analyzer.py
git commit -m "feat: analyze sources for wiki ingest"
```

---

### Task 7: 写入 Source/Concept/Topic Wiki 页面

**Files:**
- Create: `src/expert_digest/wiki/writer.py`
- Test: `tests/test_wiki_writer.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_wiki_writer.py`：

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_wiki_writer.py -q`

Expected: FAIL，错误包含 `No module named 'expert_digest.wiki.writer'`。

- [ ] **Step 3: 实现 writer**

创建 `src/expert_digest/wiki/writer.py`：

```python
"""Write source analysis results into a Markdown wiki vault."""

from __future__ import annotations

import re
from dataclasses import dataclass

from expert_digest.domain.models import EvidenceSpan
from expert_digest.wiki.models import SourceAnalysis, SourceRef, WikiPage
from expert_digest.wiki.vault import WikiVault


@dataclass(frozen=True)
class WikiWriteResult:
    sources: list[str]
    concepts: list[str]
    topics: list[str]


def write_analysis_to_vault(
    *,
    vault: WikiVault,
    analysis: SourceAnalysis,
    evidence_spans: list[EvidenceSpan],
) -> dict[str, list[str]]:
    source_path = f"sources/{analysis.source_id}.md"
    source_ref = SourceRef(
        source_id=analysis.source_id,
        title=analysis.source_title,
        url=analysis.url,
        evidence_span_ids=analysis.evidence_span_ids,
    )
    evidence_lookup = {span.id: span for span in evidence_spans}
    source_body = _render_source_body(
        analysis=analysis,
        evidence_lookup=evidence_lookup,
    )
    vault.write_page(
        WikiPage(
            path=source_path,
            page_type="source",
            title=analysis.source_title,
            body=source_body,
            sources=[source_ref],
            confidence=analysis.confidence,
        )
    )

    concept_paths = []
    for concept in analysis.concepts[:8]:
        path = f"concepts/{_slug(concept)}.md"
        concept_paths.append(path)
        vault.write_page(
            WikiPage(
                path=path,
                page_type="concept",
                title=concept,
                body=_render_concept_body(concept=concept, analysis=analysis),
                sources=[source_ref],
                confidence=analysis.confidence,
            )
        )

    topic_paths = []
    for topic in analysis.topics[:4]:
        path = f"topics/{_slug(topic)}.md"
        topic_paths.append(path)
        vault.write_page(
            WikiPage(
                path=path,
                page_type="topic",
                title=topic,
                body=_render_topic_body(topic=topic, analysis=analysis),
                sources=[source_ref],
                confidence=analysis.confidence,
            )
        )

    _append_index(
        vault=vault,
        analysis=analysis,
        source_path=source_path,
        concept_paths=concept_paths,
        topic_paths=topic_paths,
    )
    vault.append_log(f"- ingested source `{analysis.source_id}`: {analysis.source_title}")
    return {
        "sources": [source_path],
        "concepts": concept_paths,
        "topics": topic_paths,
    }


def _render_source_body(
    *,
    analysis: SourceAnalysis,
    evidence_lookup: dict[str, EvidenceSpan],
) -> str:
    lines = [f"# {analysis.source_title}", "", "## 摘要", "", analysis.summary, ""]
    lines.extend(["## 核心判断", ""])
    for claim in analysis.key_claims:
        lines.append(f"- {claim}")
    lines.extend(["", "## 证据片段", ""])
    for span_id in analysis.evidence_span_ids:
        span = evidence_lookup.get(span_id)
        if span is not None:
            lines.append(f"- `{span.id}` {span.text}")
    return "\n".join(lines).rstrip()


def _render_concept_body(*, concept: str, analysis: SourceAnalysis) -> str:
    claims = "\n".join(f"- {claim}" for claim in analysis.key_claims[:3]) or "- 暂无核心判断"
    return (
        f"# {concept}\n\n"
        f"## 来源\n\n- [[{analysis.source_title}]]\n\n"
        f"## 相关判断\n\n{claims}\n"
    ).rstrip()


def _render_topic_body(*, topic: str, analysis: SourceAnalysis) -> str:
    concepts = "、".join(f"[[{concept}]]" for concept in analysis.concepts[:6])
    return (
        f"# {topic}\n\n"
        f"## 主题摘要\n\n{analysis.summary}\n\n"
        f"## 相关概念\n\n{concepts if concepts else '暂无'}\n"
    ).rstrip()


def _append_index(
    *,
    vault: WikiVault,
    analysis: SourceAnalysis,
    source_path: str,
    concept_paths: list[str],
    topic_paths: list[str],
) -> None:
    index_path = vault.root / "index.md"
    text = index_path.read_text(encoding="utf-8") if index_path.exists() else "# Index\n"
    additions = [
        "",
        f"## {analysis.source_title}",
        "",
        f"- Source: [{analysis.source_title}]({source_path})",
    ]
    for path in topic_paths:
        additions.append(f"- Topic: [{path.removeprefix('topics/').removesuffix('.md')}]({path})")
    for path in concept_paths[:5]:
        additions.append(f"- Concept: [{path.removeprefix('concepts/').removesuffix('.md')}]({path})")
    index_path.write_text(text.rstrip() + "\n" + "\n".join(additions) + "\n", encoding="utf-8")


def _slug(value: str) -> str:
    compact = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "-", value).strip("-")
    return compact.lower() or "untitled"
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_wiki_writer.py -q`

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add src/expert_digest/wiki/writer.py tests/test_wiki_writer.py
git commit -m "feat: write analysis into wiki vault"
```

---

### Task 8: Wiki-native 检索与来源回溯

**Files:**
- Create: `src/expert_digest/wiki/retriever.py`
- Test: `tests/test_wiki_retriever.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_wiki_retriever.py`：

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_wiki_retriever.py -q`

Expected: FAIL，错误包含 `No module named 'expert_digest.wiki.retriever'`。

- [ ] **Step 3: 扩展模型**

在 `src/expert_digest/wiki/models.py` 添加：

```python
@dataclass(frozen=True)
class WikiSearchHit:
    page: WikiPage
    score: float
    matched_terms: list[str]
    source_ids: list[str]
```

- [ ] **Step 4: 实现检索**

创建 `src/expert_digest/wiki/retriever.py`：

```python
"""Wiki-native retrieval over Markdown pages."""

from __future__ import annotations

import re

from expert_digest.wiki.models import WikiSearchHit
from expert_digest.wiki.vault import WikiVault


def search_wiki(
    *,
    vault: WikiVault,
    query: str,
    top_k: int = 5,
) -> list[WikiSearchHit]:
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    terms = _tokenize(query)
    if not terms:
        return []

    hits: list[WikiSearchHit] = []
    for page in vault.list_pages():
        title = page.title.lower()
        body = page.body.lower()
        matched: list[str] = []
        score = 0.0
        for term in terms:
            lowered = term.lower()
            if lowered in title:
                score += 3.0
                matched.append(term)
            if lowered in body:
                score += 1.0
                if term not in matched:
                    matched.append(term)
        if score <= 0.0:
            continue
        hits.append(
            WikiSearchHit(
                page=page,
                score=score,
                matched_terms=matched,
                source_ids=[source.source_id for source in page.sources],
            )
        )
    return sorted(
        hits,
        key=lambda item: (-item.score, item.page.page_type, item.page.title),
    )[:top_k]


def _tokenize(text: str) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []
    latin = re.findall(r"[A-Za-z0-9_+-]{2,}", normalized)
    cjk = re.findall(r"[\u4e00-\u9fff]{2,}", normalized)
    if latin or cjk:
        return latin + cjk
    return [normalized]
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_wiki_retriever.py -q`

Expected: PASS。

- [ ] **Step 6: 提交**

```powershell
git add src/expert_digest/wiki/models.py src/expert_digest/wiki/retriever.py tests/test_wiki_retriever.py
git commit -m "feat: add wiki-native retrieval"
```

---

### Task 9: Wiki 质量评估报告

**Files:**
- Create: `src/expert_digest/wiki/evaluator.py`
- Test: `tests/test_wiki_evaluator.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_wiki_evaluator.py`：

```python
from expert_digest.wiki.evaluator import evaluate_wiki
from expert_digest.wiki.models import SourceRef, WikiPage
from expert_digest.wiki.vault import WikiVault


def test_evaluate_wiki_reports_traceability_and_coverage(tmp_path):
    vault = WikiVault(root=tmp_path / "wiki" / "huang")
    vault.initialize(expert_id="huang", expert_name="黄彦臻", purpose="沉淀公开文章。")
    vault.write_page(
        WikiPage(
            path="sources/doc-1.md",
            page_type="source",
            title="泡泡玛特复盘",
            body="## 摘要\n\n有来源。",
            sources=[SourceRef(source_id="doc-1", title="泡泡玛特复盘", evidence_span_ids=["span-1"])],
        )
    )
    vault.write_page(
        WikiPage(
            path="topics/ip.md",
            page_type="topic",
            title="IP 运营",
            body="## 判断\n\n有来源。",
            sources=[SourceRef(source_id="doc-1", title="泡泡玛特复盘", evidence_span_ids=["span-1"])],
        )
    )

    report = evaluate_wiki(vault=vault, expected_source_count=1)

    assert report.page_count >= 2
    assert report.source_page_count == 1
    assert report.pages_with_sources >= 2
    assert report.traceability_ratio == 1.0
    assert report.coverage_ratio == 1.0


def test_evaluate_wiki_detects_pages_without_sources(tmp_path):
    vault = WikiVault(root=tmp_path / "wiki" / "huang")
    vault.initialize(expert_id="huang", expert_name="黄彦臻", purpose="沉淀公开文章。")
    vault.write_page(
        WikiPage(path="topics/no-source.md", page_type="topic", title="无来源", body="没有来源。")
    )

    report = evaluate_wiki(vault=vault, expected_source_count=1)

    assert "topics/no-source.md" in report.pages_missing_sources
    assert report.traceability_ratio < 1.0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_wiki_evaluator.py -q`

Expected: FAIL，错误包含 `No module named 'expert_digest.wiki.evaluator'`。

- [ ] **Step 3: 扩展模型**

在 `src/expert_digest/wiki/models.py` 添加：

```python
@dataclass(frozen=True)
class WikiQualityReport:
    page_count: int
    source_page_count: int
    pages_with_sources: int
    pages_missing_sources: list[str]
    traceability_ratio: float
    coverage_ratio: float
```

- [ ] **Step 4: 实现 evaluator**

创建 `src/expert_digest/wiki/evaluator.py`：

```python
"""Wiki quality evaluation helpers."""

from __future__ import annotations

from expert_digest.wiki.models import WikiQualityReport
from expert_digest.wiki.vault import WikiVault


def evaluate_wiki(
    *,
    vault: WikiVault,
    expected_source_count: int,
) -> WikiQualityReport:
    if expected_source_count < 0:
        raise ValueError("expected_source_count must be >= 0")
    pages = [
        page
        for page in vault.list_pages()
        if page.page_type not in {"unknown"} and page.path not in {"purpose.md", "schema.md", "index.md", "log.md"}
    ]
    source_pages = [page for page in pages if page.page_type == "source"]
    pages_with_sources = [page for page in pages if page.sources]
    missing = [page.path for page in pages if not page.sources]
    traceability_ratio = (
        round(len(pages_with_sources) / len(pages), 4) if pages else 1.0
    )
    coverage_ratio = (
        round(len(source_pages) / expected_source_count, 4)
        if expected_source_count > 0
        else 1.0
    )
    if coverage_ratio > 1.0:
        coverage_ratio = 1.0
    return WikiQualityReport(
        page_count=len(pages),
        source_page_count=len(source_pages),
        pages_with_sources=len(pages_with_sources),
        pages_missing_sources=missing,
        traceability_ratio=traceability_ratio,
        coverage_ratio=coverage_ratio,
    )
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_wiki_evaluator.py -q`

Expected: PASS。

- [ ] **Step 6: 提交**

```powershell
git add src/expert_digest/wiki/models.py src/expert_digest/wiki/evaluator.py tests/test_wiki_evaluator.py
git commit -m "feat: evaluate wiki quality"
```

---

### Task 10: CLI 打通 Wiki Foundation 流程

**Files:**
- Modify: `src/expert_digest/cli.py`
- Test: `tests/test_cli_wiki.py`
- Modify: `README.md`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_cli_wiki.py`：

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_cli_wiki.py -q`

Expected: FAIL，CLI 不支持 `build-evidence`。

- [ ] **Step 3: 修改 CLI imports**

在 `src/expert_digest/cli.py` 添加 imports：

```python
from expert_digest.processing.evidence_builder import build_document_evidence
from expert_digest.wiki.analyzer import analyze_document_evidence
from expert_digest.wiki.evaluator import evaluate_wiki
from expert_digest.wiki.retriever import search_wiki
from expert_digest.wiki.vault import WikiVault
from expert_digest.wiki.writer import write_analysis_to_vault
```

从 storage import 添加：

```python
clear_evidence,
list_evidence_spans,
list_parent_sections,
save_evidence_spans,
save_parent_sections,
```

- [ ] **Step 4: 添加 CLI command handlers**

在 `main()` 中 parser help 前添加：

```python
    if args.command == "build-evidence":
        documents = list_documents(args.db)
        removed = clear_evidence(args.db) if args.rebuild else {
            "parent_sections": 0,
            "chunks": 0,
            "evidence_spans": 0,
        }
        all_sections = []
        all_chunks = []
        all_spans = []
        for document in documents:
            evidence = build_document_evidence(
                clean_document(document),
                parent_max_chars=args.parent_max_chars,
                child_max_chars=args.child_max_chars,
                child_min_chars=args.child_min_chars,
                span_max_chars=args.span_max_chars,
            )
            all_sections.extend(evidence.parent_sections)
            all_chunks.extend(evidence.chunks)
            all_spans.extend(evidence.evidence_spans)
        save_parent_sections(args.db, all_sections)
        save_chunks(args.db, all_chunks)
        save_evidence_spans(args.db, all_spans)
        print(
            "Built evidence: "
            f"documents={len(documents)} sections={len(all_sections)} "
            f"chunks={len(all_chunks)} spans={len(all_spans)} "
            f"cleared={removed}"
        )
        return 0

    if args.command == "build-wiki":
        vault = WikiVault(root=args.wiki_root)
        vault.initialize(
            expert_id=args.expert_id,
            expert_name=args.expert_name,
            purpose=args.purpose,
        )
        documents = list_documents(args.db)
        sections = list_parent_sections(args.db)
        spans = list_evidence_spans(args.db)
        chunks = list_chunks(args.db)
        sections_by_document = {}
        for section in sections:
            sections_by_document.setdefault(section.document_id, []).append(section)
        chunks_by_document = {}
        for chunk in chunks:
            chunks_by_document.setdefault(chunk.document_id, []).append(chunk)
        spans_by_document = {}
        for span in spans:
            spans_by_document.setdefault(span.document_id, []).append(span)
        written_sources = 0
        for document in documents:
            evidence = _document_evidence_from_store(
                document=document,
                sections=sections_by_document.get(document.id, []),
                chunks=chunks_by_document.get(document.id, []),
                spans=spans_by_document.get(document.id, []),
            )
            analysis = analyze_document_evidence(evidence)
            write_analysis_to_vault(
                vault=vault,
                analysis=analysis,
                evidence_spans=evidence.evidence_spans,
            )
            written_sources += 1
        print(f"Built wiki: sources={written_sources} root={args.wiki_root}")
        return 0

    if args.command == "search-wiki":
        hits = search_wiki(
            vault=WikiVault(root=args.wiki_root),
            query=args.query,
            top_k=args.top_k,
        )
        for hit in hits:
            print(
                f"score={hit.score:.2f}\t{hit.page.page_type}\t"
                f"{hit.page.title}\t{hit.page.path}\t"
                f"sources={','.join(hit.source_ids)}"
            )
        return 0

    if args.command == "eval-wiki":
        report = evaluate_wiki(
            vault=WikiVault(root=args.wiki_root),
            expected_source_count=args.expected_source_count,
        )
        _print_json_safely(asdict(report))
        return 0
```

在文件底部 helper 区域新增：

```python
def _document_evidence_from_store(
    *,
    document,
    sections,
    chunks,
    spans,
):
    from expert_digest.processing.evidence_builder import DocumentEvidence

    return DocumentEvidence(
        document=document,
        parent_sections=sections,
        chunks=chunks,
        evidence_spans=spans,
    )
```

- [ ] **Step 5: 添加 parser 命令**

在 `_build_parser()` 添加：

```python
    evidence_parser = subparsers.add_parser("build-evidence")
    evidence_parser.add_argument("--db", type=Path, default=DEFAULT_DATABASE_PATH)
    evidence_parser.add_argument("--parent-max-chars", type=int, default=2400)
    evidence_parser.add_argument("--child-max-chars", type=int, default=700)
    evidence_parser.add_argument("--child-min-chars", type=int, default=80)
    evidence_parser.add_argument("--span-max-chars", type=int, default=220)
    evidence_parser.add_argument("--rebuild", action="store_true")

    wiki_parser = subparsers.add_parser("build-wiki")
    wiki_parser.add_argument("--db", type=Path, default=DEFAULT_DATABASE_PATH)
    wiki_parser.add_argument("--wiki-root", type=Path, default=Path("data/wiki/default"))
    wiki_parser.add_argument("--expert-id", default="default")
    wiki_parser.add_argument("--expert-name", default="unknown")
    wiki_parser.add_argument("--purpose", default="沉淀专家公开内容。")

    search_wiki_parser = subparsers.add_parser("search-wiki")
    search_wiki_parser.add_argument("query")
    search_wiki_parser.add_argument("--wiki-root", type=Path, default=Path("data/wiki/default"))
    search_wiki_parser.add_argument("--top-k", type=int, default=5)

    eval_wiki_parser = subparsers.add_parser("eval-wiki")
    eval_wiki_parser.add_argument("--wiki-root", type=Path, default=Path("data/wiki/default"))
    eval_wiki_parser.add_argument("--expected-source-count", type=int, default=0)
```

- [ ] **Step 6: 运行 CLI 测试**

Run: `python -m pytest tests/test_cli_wiki.py -q`

Expected: PASS。

- [ ] **Step 7: 更新 README**

在 `README.md` 的使用说明中添加：

```markdown
Build hierarchical evidence for Wiki Foundation:

```powershell
expert-digest build-evidence --db data/processed/zhihu_huang.sqlite3 --rebuild
```

Build an Expert Wiki vault:

```powershell
expert-digest build-wiki --db data/processed/zhihu_huang.sqlite3 --wiki-root data/wiki/huang --expert-id huang --expert-name "黄彦臻" --purpose "沉淀黄彦臻公开文章中的投资分析框架。"
```

Search the wiki:

```powershell
expert-digest search-wiki "泡泡玛特 核心能力" --wiki-root data/wiki/huang
```

Evaluate wiki quality:

```powershell
expert-digest eval-wiki --wiki-root data/wiki/huang --expected-source-count 824
```
```

- [ ] **Step 8: 运行回归测试**

Run: `python -m pytest -q`

Expected: 全部通过。

- [ ] **Step 9: 提交**

```powershell
git add src/expert_digest/cli.py README.md tests/test_cli_wiki.py
git commit -m "feat: add wiki foundation cli"
```

---

### Task 11: 文档与状态更新

**Files:**
- Modify: `docs/progress_status.md`
- Create: `docs/wiki_foundation_usage.md`

- [ ] **Step 1: 创建使用文档**

创建 `docs/wiki_foundation_usage.md`：

```markdown
# ExpertDigest Wiki Foundation Usage

## 目标

Wiki Foundation 把专家原文编译成 Markdown vault。它是 handbook、skill 和问答升级的中间知识层。

## 推荐流程

```powershell
expert-digest import-zhihu "D:\Project\Zhihu_Crawler\data\zhihu\huang-wei-yan-30" --db data/processed/zhihu_huang.sqlite3
expert-digest build-evidence --db data/processed/zhihu_huang.sqlite3 --rebuild
expert-digest build-wiki --db data/processed/zhihu_huang.sqlite3 --wiki-root data/wiki/huang --expert-id huang --expert-name "黄彦臻" --purpose "沉淀黄彦臻公开文章中的投资分析框架。"
expert-digest search-wiki "泡泡玛特 核心能力" --wiki-root data/wiki/huang
expert-digest eval-wiki --wiki-root data/wiki/huang --expected-source-count 824
```

## Vault 结构

```text
data/wiki/<expert_id>/
  purpose.md
  schema.md
  index.md
  log.md
  sources/
  concepts/
  topics/
  theses/
  reviews/
```

## 质量标准

- 每个 source 都应有 `sources/<source_id>.md`。
- 每个生成页面应有 frontmatter。
- 核心页面应带 source refs。
- `eval-wiki` 应输出 traceability 和 coverage。
```

- [ ] **Step 2: 更新进度文档**

在 `docs/progress_status.md` 增加 Wiki Foundation 状态：

```markdown
## ExpertDigest 2.0 Wiki Foundation

- Design: completed (`docs/superpowers/specs/2026-04-21-expertdigest-2-wiki-upgrade-design.md`)
- Implementation plan: completed (`docs/superpowers/plans/2026-04-21-expertdigest-2-wiki-foundation.md`)
- First implementation target: hierarchical evidence model + Markdown wiki vault + deterministic ingest baseline
```

- [ ] **Step 3: 运行文档相关测试**

Run: `python -m pytest tests/test_import.py -q`

Expected: PASS。

- [ ] **Step 4: 提交**

```powershell
git add docs/progress_status.md docs/wiki_foundation_usage.md
git commit -m "docs: document wiki foundation usage"
```

---

### Task 12: 最终验收

**Files:**
- No code changes unless verification exposes a concrete bug.

- [ ] **Step 1: 完整测试**

Run: `python -m pytest -q`

Expected: 全部测试通过。

- [ ] **Step 2: Ruff 检查**

Run: `python -m ruff check .`

Expected: `All checks passed!`

- [ ] **Step 3: 一条样例链路验证**

Run:

```powershell
expert-digest import-jsonl data/sample/articles.jsonl --db data/processed/wiki_foundation_sample.sqlite3
expert-digest build-evidence --db data/processed/wiki_foundation_sample.sqlite3 --rebuild
expert-digest build-wiki --db data/processed/wiki_foundation_sample.sqlite3 --wiki-root data/wiki/sample --expert-id sample --expert-name "样例专家" --purpose "验证 Wiki Foundation 样例链路。"
expert-digest search-wiki "核心能力" --wiki-root data/wiki/sample
expert-digest eval-wiki --wiki-root data/wiki/sample --expected-source-count 3
```

Expected:

- `build-evidence` 输出 `Built evidence`。
- `build-wiki` 输出 `Built wiki`。
- `search-wiki` 至少输出 1 行 hit。
- `eval-wiki` 输出 JSON，包含 `traceability_ratio` 和 `coverage_ratio`。

- [ ] **Step 4: 检查工作区**

Run: `git status --short`

Expected: 只允许出现样例运行生成的数据文件。如果出现 `data/processed/wiki_foundation_sample.sqlite3` 或 `data/wiki/sample/`，确认它们不应提交，保留为本地验证产物。

- [ ] **Step 5: 更新最终进度**

若 Task 12 验证通过，在 `docs/progress_status.md` 添加验证日期、命令和结果：

```markdown
## Wiki Foundation Verification

- Date: 2026-04-21
- Commands:
  - `python -m pytest -q`
  - `python -m ruff check .`
  - sample import -> build-evidence -> build-wiki -> search-wiki -> eval-wiki
- Result: passed
```

- [ ] **Step 6: 提交最终状态**

```powershell
git add docs/progress_status.md
git commit -m "docs: record wiki foundation verification"
```

---

## 自查清单

- Spec 覆盖：本计划覆盖证据模型、Wiki vault、两步 ingest、Wiki 检索、质量评估、CLI 暴露。
- 第一阶段不绑定向量数据库：本计划没有引入 Qdrant、LanceDB、Chroma、pgvector。
- 框架边界：本计划没有引入 LlamaIndex、LangChain、LangGraph。
- Handbook/Skill：本计划只建立它们的 Wiki 基础，不在第一阶段重写生成器。
- 质量门：Task 9 和 Task 12 提供 Wiki 质量报告与完整验证。
