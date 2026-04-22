# ExpertDigest 2.0 Wiki-First Upgrade Design

Date: 2026-04-21

## 1. Purpose

ExpertDigest 2.0 upgrades the project from a baseline raw-chunk RAG demo into a
wiki-first expert knowledge system.

The first upgrade phase focuses on building an Expert Wiki knowledge base. The
wiki becomes the durable middle layer between raw expert sources and downstream
applications such as handbook generation, skill generation, and question
answering.

The core product shift is:

```text
Raw source RAG -> Expert Wiki + source backtrace
```

The system should not rely on top-k raw chunks as the primary source for
handbook and skill generation. Instead, raw sources are compiled into persistent,
reviewable, source-grounded wiki pages. Applications then use the wiki first and
trace claims back to raw evidence when needed.

## 2. Confirmed Design Decisions

1. The first phase's primary artifact is the Expert Wiki knowledge base.
2. The wiki uses a Markdown vault with YAML frontmatter and wikilinks.
3. The first phase does not bind the project to Qdrant or any other vector
   database.
4. Retrieval is exposed through pluggable evidence interfaces.
5. The evidence model is hierarchical:

```text
SourceDocument -> ParentSection -> ChildChunk -> EvidenceSpan
```

6. Core architecture remains project-owned. LlamaIndex, LangChain, vector
   databases, and similar tools can be added behind adapters, but they do not
   own the system's domain model or artifact logic.
7. LangGraph is out of scope for the first phase.
8. Wiki ingest uses a two-step flow:

```text
Analyze Source -> Write/Update Wiki Pages
```

9. The first phase includes explicit wiki quality evaluation.

## 3. Non-Goals

The first phase will not:

- Build an enterprise knowledge platform.
- Add multi-user permissions or tenant management.
- Rebuild the Streamlit UI.
- Make Qdrant, LanceDB, Chroma, pgvector, or any other vector store mandatory.
- Introduce GraphRAG or Neo4j.
- Introduce a multi-agent architecture.
- Treat the generated Skill as a compressed copy of the source articles.
- Replace source traceability with model memory.

## 4. Target Architecture

```text
Raw Sources
  -> Source Processing
  -> Hierarchical Evidence Model
  -> Expert Wiki
  -> Skill / Handbook / QA / MCP Tools
```

### 4.1 Raw Source Layer

Stores source documents as the truth layer. A source can be a JSONL article,
Markdown document, Zhihu export item, or future imported document type.

Raw sources are not overwritten by generated interpretations.

### 4.2 Source Processing Layer

Transforms raw source text into structured evidence units. This layer replaces
the current single-level character chunking strategy.

Responsibilities:

- Normalize text.
- Preserve document metadata.
- Detect headings and logical sections where possible.
- Build parent sections.
- Build child chunks for retrieval.
- Extract evidence spans for claim citation.

### 4.3 Evidence Model

The new evidence model has four levels.

| Level | Purpose |
| --- | --- |
| `SourceDocument` | Full article or source item, including author, title, URL, source, and created date. |
| `ParentSection` | Larger context unit used to give the model enough surrounding argument structure. |
| `ChildChunk` | Smaller retrieval unit used for precise matching. |
| `EvidenceSpan` | Minimum citation unit used to support a claim and trace back to source text. |

The model is designed so retrieval can find precise child chunks while generation
receives enough parent context and cites specific spans.

### 4.4 Expert Wiki Layer

The Expert Wiki is a Markdown vault:

```text
data/wiki/<expert_id>/
  purpose.md
  schema.md
  index.md
  log.md
  sources/
    <source_id>.md
  concepts/
    <concept_slug>.md
  topics/
    <topic_slug>.md
  theses/
    <thesis_slug>.md
  reviews/
    <review_id>.md
```

All generated pages use YAML frontmatter. A topic page example:

```yaml
---
type: topic
title: 泡泡玛特的核心能力
sources:
  - source_id: zhihu_article_2023428236160280283
    url: http://zhuanlan.zhihu.com/p/2023428236160280283
confidence: medium
updated_at: 2026-04-21
---
```

Page links use `[[wikilink]]` syntax so the wiki can be read by humans and parsed
by the system.

### 4.5 Application Layer

Current CLI and MCP capabilities remain, but their future source of knowledge
changes.

Future tools should include:

- `read_wiki_page`
- `search_wiki`
- `search_evidence`
- `trace_claim_sources`
- `generate_handbook`
- `generate_skill`
- `ask_author`
- `deep_answer`
- `lint_wiki`

## 5. Query Strategy

ExpertDigest 2.0 should not use one RAG path for every question. It should route
questions by intent.

| Query Type | Primary Path |
| --- | --- |
| Method transfer | Skill + Wiki |
| Expert factual claim | Wiki + source backtrace |
| Source citation request | EvidenceSpan / SourceDocument lookup |
| Deep synthesis | Wiki + multi-step evidence retrieval + synthesis |
| Missing evidence | Refuse or ask for more source material |

The normal answer path is:

```text
Question
  -> classify intent
  -> read Expert Skill for answer constraints
  -> search wiki pages / source summaries / citation index
  -> trace claims back to evidence spans
  -> optionally retrieve raw source context
  -> generate source-grounded answer
```

This means retrieval remains necessary, but it is not always raw-chunk vector
RAG. Retrieval can operate over wiki pages, metadata, wikilinks, source refs,
BM25/full text, or later semantic vector indexes.

## 6. Retrieval and Vector Database Position

The first phase does not choose a mandatory vector database.

Instead, it defines retrieval interfaces:

```text
WikiRetriever
EvidenceRetriever
SemanticRetriever
```

Initial implementations can use:

- Markdown vault parsing.
- Frontmatter filters.
- Title and slug lookup.
- Wikilink graph expansion.
- Source reference backtrace.
- BM25 or other local full-text search.

Vector databases remain optional backend adapters. Candidate future backends
include Qdrant, LanceDB, pgvector, Chroma, Weaviate, and Milvus.

Selection should be driven by evaluation:

- Choose Qdrant if dense/sparse hybrid search, payload filtering, and multi-stage
  retrieval become important.
- Choose LanceDB if local file-based operation and Python-native workflow matter
  most.
- Choose pgvector if the project adopts Postgres as the main metadata store.
- Choose Chroma only if simplicity outweighs advanced retrieval control.
- Choose Milvus only if scale justifies the operational weight.

## 7. Source Ingest Flow

Wiki ingest is a two-step process.

### 7.1 Analyze Source

Inputs:

- `SourceDocument`
- `ParentSection` records
- Existing wiki index and candidate related pages

Outputs:

- Source summary
- Key claims
- Key concepts
- Entities
- Topics
- Thesis candidates
- Evidence spans
- Related wiki pages
- Contradictions or review notes

### 7.2 Write or Update Wiki Pages

The writer updates:

- `sources/<source_id>.md`
- `concepts/<concept_slug>.md`
- `topics/<topic_slug>.md`
- `theses/<thesis_slug>.md`
- `reviews/<review_id>.md`
- `index.md`
- `log.md`

All core claims must include source references. If the system cannot support a
claim with source refs, it should mark the claim as uncertain or create a review
item instead of silently writing it as fact.

## 8. Handbook and Skill Generation

Handbook generation should consume the wiki, not raw top-k chunks.

The handbook generator should read:

- `index.md`
- topic pages
- thesis pages
- source summaries
- review notes
- source refs for core claims

Skill generation should also consume the wiki.

The Skill should contain:

- Expert analysis habits
- Recurring concepts
- Reasoning patterns
- Question handling rules
- Citation rules
- Refusal rules
- Boundaries and uncertainty handling

The Skill should not contain a large factual compression of all source content.
Facts remain in the wiki and source evidence layer.

## 9. Evaluation

First-phase acceptance depends on wiki quality, not just command success.

Evaluation dimensions:

1. Source Traceability
   - Every generated wiki page has source refs.
   - Core claims can trace to an `EvidenceSpan`.

2. Coverage
   - Every ingested source has a source summary page.
   - Recurring concepts and topics are represented in the wiki.

3. Faithfulness
   - Generated claims are supported by cited evidence.
   - Unsupported claims are marked uncertain or sent to review.

4. Usability
   - Handbook sections generated from the wiki are readable and useful.
   - Skill output contains executable rules, not generic style descriptions.

5. Query Answerability
   - A small evaluation set can be answered through Wiki + source backtrace.
   - Answers include citations or refuse when evidence is missing.

## 10. Reference Projects

### 10.1 WeKnora

ExpertDigest should borrow WeKnora's engineering ideas:

- Quick answer vs intelligent reasoning modes.
- Parent-child chunking.
- Hybrid retrieval.
- Reranking.
- MCP tool exposure.
- End-to-end retrieval/generation tests.
- Pluggable retrieval backend design.

ExpertDigest should not copy WeKnora's enterprise platform scope in the first
phase.

### 10.2 llm_wiki

ExpertDigest should borrow llm_wiki's knowledge architecture:

- Raw sources / wiki / schema separation.
- Ingest / query / lint operations.
- Markdown pages with frontmatter.
- `index.md`, `purpose.md`, `schema.md`, and `log.md`.
- Wikilinks.
- Source references.
- Review pages.

Direct source-code reuse is avoided because llm_wiki is GPLv3 and has a
different Tauri/Rust/React architecture.

## 11. Open Implementation Questions

These are deferred to the implementation plan:

- Exact page schemas for each wiki page type.
- Exact `EvidenceSpan` location fields.
- Whether first-phase full-text search uses SQLite FTS5, a pure Python BM25
  package, or a small local search index.
- Whether a first semantic retriever adapter should be included as an optional
  experiment.
- How many evaluation examples are required for the first quality gate.

## 12. First Phase Acceptance Criteria

The first phase is complete when:

1. A source import can produce the hierarchical evidence model.
2. The wiki vault can be initialized for one expert.
3. Ingest can generate or update source, concept, topic, thesis, index, and log
   pages.
4. Every generated wiki page records source references.
5. A wiki query can return relevant pages and trace key claims to source
   evidence.
6. A basic handbook generated from the wiki is materially better than the
   previous chunk-based handbook.
7. A basic skill generated from the wiki contains concrete reasoning and citation
   rules.
8. A first wiki quality report covers traceability, coverage, faithfulness,
   usability, and query answerability.
