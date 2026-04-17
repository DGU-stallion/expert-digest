# ExpertDigest Progress Status (2026-04-16)

## Current Milestone State

- M0 `project init`: completed
- M1 `data foundation`: completed
- M2 `data processing`: in progress (chunk pipeline + local embedding + vector retrieval delivered)

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

## Live Data Integration

Zhihu dataset imported from:

`D:\Project\Zhihu_Crawler\data\zhihu\huang-wei-yan-30`

Output database:

`data/processed/zhihu_huang.sqlite3`

Current data snapshot:

- documents: 824
- chunks: 864
- authors: 1 (`黄彦臻`)

## Next Recommended Steps (M2 -> M3)

1. Add text cleaner (`processing/cleaner.py`) and re-run chunk pipeline.
2. Add embedding generation and vector store abstraction.
3. Implement basic retriever (`top_k`) and evidence object.
4. Start M3 baseline RAG answer command with citation output.
