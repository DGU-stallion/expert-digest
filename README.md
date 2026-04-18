# ExpertDigest

ExpertDigest is a local expert-content knowledge distillation project. The MVP
will focus on importing local expert articles, storing them, retrieving evidence,
answering questions, recommending original texts, and generating a Markdown
learning handbook.

This repository currently includes:

- M1 data foundation: local JSONL/Markdown article import, SQLite document
  storage, and author-based document listing.
- M2 data processing: cleaner/splitter pipeline, chunk persistence, and local
  embeddings.
- M3 baseline RAG: evidence retrieval, structured answer output, and refusal
  policy for low-confidence/no-evidence questions.
- M4 handbook generation: hybrid (LLM + deterministic fallback) handbook
  generation with JSON observability metadata export.
- M5 streamlit demo: import/process/ask/handbook flow for local walkthrough.
- M6 topic clustering enhancement: cluster output, report metrics, and topic
  naming strategy (deterministic + optional LLM fallback).
- M7 author profile + skill baseline: profile extraction and skill draft
  generation.
- M8 MCP server baseline: Cherry Studio-ready MCP tool endpoints (stdio/sse).

## Requirements

- Python 3.11+
- Git

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project with development tools:

```powershell
python -m pip install -e ".[dev]"
```

## Verify

Run tests:

```powershell
python -m pytest
```

Run lint checks:

```powershell
python -m ruff check .
```

Verify the package import:

```powershell
python -c "import expert_digest; print(expert_digest.__version__)"
```

## M1 Usage

Import the sample JSONL articles:

```powershell
expert-digest import-jsonl data/sample/articles.jsonl --db data/processed/expert_digest.sqlite3
```

List every imported document:

```powershell
expert-digest list-documents --db data/processed/expert_digest.sqlite3
```

Filter by author:

```powershell
expert-digest list-documents --author "陈一鸣" --db data/processed/expert_digest.sqlite3
```

Import a folder of Markdown files:

```powershell
expert-digest import-markdown path/to/markdown-folder --db data/processed/expert_digest.sqlite3
```

Import a Zhihu crawler export folder (expects `index/content_index.jsonl`):

```powershell
expert-digest import-zhihu "D:\Project\Zhihu_Crawler\data\zhihu\huang-wei-yan-30" --db data/processed/zhihu_huang.sqlite3
```

Build chunks from imported documents:

```powershell
expert-digest build-chunks --db data/processed/zhihu_huang.sqlite3 --max-chars 1200 --min-chars 80
```

Rebuild chunks (clear old chunks then regenerate):

```powershell
expert-digest rebuild-chunks --db data/processed/zhihu_huang.sqlite3 --max-chars 1200 --min-chars 80
```

Build chunk embeddings in SQLite:

```powershell
expert-digest build-embeddings --db data/processed/zhihu_huang.sqlite3
```

Rebuild embeddings from existing chunks:

```powershell
expert-digest rebuild-embeddings --db data/processed/zhihu_huang.sqlite3
```

Search top chunks by semantic similarity:

```powershell
expert-digest search-chunks "泡泡玛特 IP 运营" --db data/processed/zhihu_huang.sqlite3 --top-k 5
```

Ask a question with structured RAG output:

```powershell
expert-digest ask "泡泡玛特的核心能力是什么？" --db data/processed/zhihu_huang.sqlite3 --top-k 3
```

Ask with machine-readable JSON:

```powershell
expert-digest ask "泡泡玛特的核心能力是什么？" --db data/processed/zhihu_huang.sqlite3 --format json
```

Generate handbook (hybrid mode, default):

```powershell
expert-digest generate-handbook --db data/processed/zhihu_huang.sqlite3 --output data/outputs/handbook.md
```

Generate handbook organized by clustered topics:

```powershell
expert-digest generate-handbook --db data/processed/zhihu_huang.sqlite3 --theme-source cluster --num-topics 4
```

Generate handbook as JSON result (with runtime metadata):

```powershell
expert-digest generate-handbook --db data/processed/zhihu_huang.sqlite3 --format json
```

Generate handbook and save run metadata to JSON file:

```powershell
expert-digest generate-handbook --db data/processed/zhihu_huang.sqlite3 --format json --save-run-metadata data/outputs/handbook_run_metadata.json
```

Force deterministic mode (no LLM):

```powershell
expert-digest generate-handbook --db data/processed/zhihu_huang.sqlite3 --synthesis-mode deterministic
```

Cluster topics from local chunk embeddings:

```powershell
expert-digest cluster-topics --db data/processed/zhihu_huang.sqlite3 --num-topics 3 --top-docs 2
```

Cluster topics with optional LLM naming (fallback to deterministic labels when
LLM is unavailable or fails):

```powershell
expert-digest cluster-topics --db data/processed/zhihu_huang.sqlite3 --num-topics 3 --top-docs 2 --label-mode llm --format json
```

Export cluster report artifact (topic distribution + similarity proxy metrics):

```powershell
expert-digest cluster-topics --db data/processed/zhihu_huang.sqlite3 --num-topics 3 --top-docs 2 --format json --report-output data/outputs/topic_report.json
```

Run the M5 Streamlit demo (import/process/ask/handbook preview):

```powershell
python -m pip install -e ".[app]"
streamlit run src/expert_digest/app/streamlit_app.py
```

Build deterministic author profile:

```powershell
expert-digest build-author-profile --db data/processed/zhihu_huang.sqlite3 --format json
```

Generate skill draft from profile:

```powershell
expert-digest generate-skill-draft --db data/processed/zhihu_huang.sqlite3 --output data/outputs/skill.md
```

Run MCP server baseline for Cherry Studio (stdio):

```powershell
python -m pip install -e ".[mcp]"
expert-digest run-mcp-server --db data/processed/zhihu_huang.sqlite3 --transport stdio
```

Cherry Studio MCP integration guide:

- `docs/m8_cherry_studio_setup.md`

In the Streamlit "导入数据" page, JSONL supports direct file upload (with
local path fallback).

JSONL input uses one article per line:

```json
{"author":"陈一鸣","title":"问题意识是学习的入口","content":"...","source":"synthetic-sample","url":"https://example.com/article","created_at":"2026-01-01"}
```

Required fields are `author`, `title`, `content`, and `source`. Optional fields
are `url` and `created_at`.

Markdown import supports optional front matter:

```markdown
---
author: 陈一鸣
title: 问题意识是学习的入口
url: https://example.com/article
created_at: 2026-01-01
---

# 问题意识是学习的入口

正文内容。
```

## GitHub Remote Setup

After creating an empty repository on GitHub, connect this local repository with
one of the following commands.

HTTPS:

```powershell
git remote add origin https://github.com/<your-user>/<your-repo>.git
git branch -M main
git push -u origin main
```

SSH:

```powershell
git remote add origin git@github.com:<your-user>/<your-repo>.git
git branch -M main
git push -u origin main
```

If `origin` already exists, replace it:

```powershell
git remote set-url origin https://github.com/<your-user>/<your-repo>.git
```
