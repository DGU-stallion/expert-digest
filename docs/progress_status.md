# ExpertDigest Progress Status (2026-04-18)

## Current Milestone State

- M0 `project init`: completed
- M1 `data foundation`: completed
- M2 `data processing`: completed
- M3 `baseline rag`: completed
- M4 `handbook generation`: completed
- M5 `streamlit demo`: completed
- M6 `topic clustering enhancement`: completed
- M7 `author profile + skill distillation baseline`: completed
- M8 `Cherry Studio MCP integration`: in progress (baseline server completed)

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
13. Hybrid mode LLM integration via local provider DB, preferring Google/Gemini config
14. Handbook runtime observability metadata (`llm_provider/model/base_url`, `latency_ms`, `fallback_used`, `error_reason`)
15. Optional run metadata export to JSON (`generate-handbook --save-run-metadata`)
16. Streamlit baseline pages: import/process/ask/handbook preview
17. Streamlit JSONL upload-first import flow with local-path fallback
18. M6 topic clustering command (`cluster-topics`) with representative documents
19. Handbook can organize sections by clustered topics (`--theme-source cluster`)
20. Topic naming strategy supports deterministic/LLM modes (`cluster-topics --label-mode`)
21. Topic cluster evaluation report with proxy metrics (`--report-output`)
22. Streamlit process page supports topic distribution and representative-doc visualization
23. Deterministic author profile extraction (`build-author-profile`)
24. Skill draft generation from author profile (`generate-skill-draft`)
25. App service support for profile/skill export (`build_author_profile_snapshot`, `generate_skill_draft`)
26. Streamlit M7 page for profile/skill preview and export
27. M8 MCP server baseline with Cherry Studio-oriented tools (`run-mcp-server`)
28. Cherry Studio integration runbook (`docs/m8_cherry_studio_setup.md`)

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

## M6 Exit Criteria Check

1. Clustering can generate topic list: pass
2. Each topic contains representative documents: pass
3. Handbook can be organized by topics: pass
4. Optional LLM naming path exists and fail-closed fallback works: pass
5. Cluster quality evaluation metrics/report output: pass

## M7 Exit Criteria Check

1. Author profile baseline (topics/keywords/reasoning patterns): pass
2. Skill draft baseline (rules/citation constraints/refusal rules): pass
3. Explicit constraint kept: factual claims still sourced from RAG evidence: pass
4. Streamlit supports profile + skill preview/export: pass

## M8 Baseline Check (in progress)

1. MCP server entry command exists (`run-mcp-server`): pass
2. Exposed tools: `ask_author`, `search_posts`, `recommend_readings`, `list_topics`, `generate_handbook`, `generate_skill`: pass
3. Uses existing retrieval/handbook/profile/skill services (no duplicate core logic): pass
4. Cherry Studio end-to-end runtime validation: pending (manual check required)

## ExpertDigest 2.0 Wiki Foundation

- Design: completed (`docs/superpowers/specs/2026-04-21-expertdigest-2-wiki-upgrade-design.md`)
- Implementation plan: completed (`docs/superpowers/plans/2026-04-21-expertdigest-2-wiki-foundation.md`)
- First implementation target: hierarchical evidence model + Markdown wiki vault + deterministic ingest baseline

## Next Recommended Steps (M8 Completion)

1. Validate MCP server end-to-end inside Cherry Studio and add setup screenshots/notes.
2. Add auth/config wrapper for multi-database routing if one workspace serves multiple experts.
3. Add MCP integration tests (tool-call contract and failure-path assertions with mock client).
