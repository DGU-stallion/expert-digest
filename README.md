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
- M5 streamlit demo (legacy): local demo UI, no longer the primary product path.
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

Install app + MCP dependencies when you need Cherry Studio MCP (and optional legacy Streamlit):

```powershell
python -m pip install -e ".[app,mcp]"
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
expert-digest list-documents --author "黄彦臻" --db data/processed/expert_digest.sqlite3
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

Generate handbook (recommended: cluster + manual taxonomy naming):

```powershell
expert-digest generate-handbook --db data/processed/zhihu_huang.sqlite3 --theme-source cluster --num-topics 8 --topic-taxonomy configs/handbook_topic_taxonomy.json --output data/outputs/huang_handbook.md
```

Hybrid mode loads LLM configuration from the local provider database by default
(`C:\Users\<you>\.cc-switch\cc-switch.db`) and prefers Google/Gemini providers
when available.

If you want to force Cherry Studio's Google Gemini 2.5 Flash (OpenAI-compatible endpoint),
set environment variables before running handbook generation:

```powershell
$env:OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
$env:OPENAI_API_KEY="your-cherry-token"
$env:OPENAI_MODEL="gemini-2.5-flash"
expert-digest generate-handbook --db data/processed/zhihu_huang.sqlite3 --author "黄彦臻" --theme-source cluster --num-topics 12 --max-themes 8 --top-k 8 --topic-taxonomy configs/handbook_topic_taxonomy.json --synthesis-mode hybrid --format json
```

Use the JSON output to confirm runtime metadata:
- `llm_enabled=true`
- `llm_model=gemini-2.5-flash`
- `fallback_used=false` (means LLM call succeeded)

Generate handbook with clustered topics and JSON metadata:

```powershell
expert-digest generate-handbook --db data/processed/zhihu_huang.sqlite3 --theme-source cluster --num-topics 8 --topic-taxonomy configs/handbook_topic_taxonomy.json --format json
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

Generate skill draft from profile:

```powershell
expert-digest build-author-profile --db data/processed/zhihu_huang.sqlite3 --format json
expert-digest generate-skill-draft --db data/processed/zhihu_huang.sqlite3 --output data/outputs/huang_skill.md
```

Run MCP server baseline for Cherry Studio (stdio):

```powershell
python -m pip install -e ".[mcp]"
expert-digest run-mcp-server --db data/processed/zhihu_huang.sqlite3 --transport stdio
```

Cherry Studio MCP integration guide:

- `docs/m8_cherry_studio_setup.md`

Run one-command local self-check (import -> process -> ask -> handbook -> profile -> skill):

```powershell
.\scripts\quickstart_selfcheck.ps1
```

JSONL input uses one article per line:

```json
{"author":"黄彦臻","title":"关于泡泡玛特的极简复盘","content":"...","source":"zhihu:article:2023428236160280283","url":"http://zhuanlan.zhihu.com/p/2023428236160280283","created_at":"2026-04-03T07:55:11.000Z"}
```

Required fields are `author`, `title`, `content`, and `source`. Optional fields
are `url` and `created_at`.

Markdown import supports optional front matter:

```markdown
---
author: 黄彦臻
title: 关于泡泡玛特的极简复盘
url: http://zhuanlan.zhihu.com/p/2023428236160280283
created_at: 2026-04-03T07:55:11.000Z
---

# 关于泡泡玛特的极简复盘

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
