# ExpertDigest Progress Status (2026-04-17)

## Current Milestone State

- M0 `project init`: completed
- M1 `data foundation`: completed
- M2 `data processing`: completed
- M3 `baseline rag`: completed
- M4 `handbook generation`: completed
- M5 `streamlit demo`: completed
- M6 `topic clustering enhancement`: in progress

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
16. Streamlit baseline pages: import/process/ask/handbook preview
17. Streamlit JSONL upload-first import flow with local-path fallback
18. M6 topic clustering command (`cluster-topics`) with representative documents
19. Handbook can organize sections by clustered topics (`--theme-source cluster`)
20. Topic naming strategy supports deterministic/LLM modes (`cluster-topics --label-mode`)

## Live Data Integration

Zhihu dataset imported from:

`D:\Project\Zhihu_Crawler\data\zhihu\huang-wei-yan-30`

Output database:

`data/processed/zhihu_huang.sqlite3`

Current data snapshot:

- documents: 824
- chunks: 864
- authors: 1 (`黄彦臻`)

## M5 Exit Criteria Check

1. Streamlit has 4 baseline pages: pass
2. Upload sample JSONL and import into SQLite: pass
3. Trigger processing and run ask loop in demo: pass
4. Preview generated handbook in UI: pass

## M6 Current Checkpoint

1. Clustering can generate topic list: pass
2. Each topic contains representative documents: pass
3. Handbook can be organized by topics: pass
4. Optional LLM naming path exists and fail-closed fallback works: pass
5. Cluster quality evaluation metrics/report output: not started

## Next Recommended Steps (M6)

1. Add lightweight cluster evaluation metrics (e.g., silhouette-like proxy or intra/inter similarity stats).
2. Export a clustering report artifact for course presentation (`data/outputs/topic_report.json/md`).
3. Add one Streamlit visualization block for topic distribution and representative docs.
