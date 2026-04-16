# ExpertDigest Progress Status (2026-04-16)

## Current Milestone State

- M0 `project init`: completed
- M1 `data foundation`: completed
- M2 `data processing`: completed
- M3 `baseline rag`: completed
- M4 `handbook generation`: completed
- M5 `streamlit demo`: not started

## Completed Capabilities

1. Import local JSONL articles into SQLite (`import-jsonl`)
2. Import local Markdown folders into SQLite (`import-markdown`)
3. Import Zhihu crawler export into SQLite (`import-zhihu`)
4. List documents and filter by author (`list-documents`)
5. Build deterministic chunks from documents (`build-chunks`)
6. Rebuild chunks from scratch (`rebuild-chunks`)
7. Save/query/clear chunks in SQLite (`chunks` table + query methods)
8. Build and rebuild chunk embeddings in SQLite (`build-embeddings`, `rebuild-embeddings`)
9. Top-k chunk similarity retrieval (`search-chunks`)
10. Structured RAG ask command with evidence output and refusal policy (`ask`)
11. JSON answer output for automation/evaluation (`ask --format json`)
12. Handbook generation in deterministic/hybrid mode (`generate-handbook`)
13. Hybrid mode LLM integration via local `ccswitch` NVIDIA provider config
14. Handbook runtime observability metadata (`llm_provider/model/base_url`, `latency_ms`, `fallback_used`, `error_reason`)
15. Optional run metadata export to JSON (`generate-handbook --save-run-metadata`)

## Live Data Integration

Zhihu dataset imported from:

`D:\Project\Zhihu_Crawler\data\zhihu\huang-wei-yan-30`

Output database:

`data/processed/zhihu_huang.sqlite3`

Current data snapshot:

- documents: 824
- chunks: 864
- authors: 1 (`黄彦臻`)

## M4 Exit Criteria Check

1. Handbook command can output Markdown: pass
2. Handbook includes required sections (overview/themes/insights/path/index): pass
3. Hybrid mode available and deterministic fallback enabled: pass
4. Runtime metadata available in JSON output and optional file export: pass

## Next Recommended Steps (M5)

1. Add Streamlit app baseline with 4 pages: import/process/ask/handbook preview.
2. Reuse existing CLI/service functions directly; avoid duplicating business logic.
3. Add a small UI smoke test checklist for demo readiness.
