# M4 Closure Report (2026-04-16)

## Scope

M4 target in the development plan:

- Generate the first readable Markdown handbook.
- Keep source traceability and section structure stable.
- Leave room for LLM integration while preserving deterministic fallback.

## Delivered

1. `generate-handbook` command supports deterministic and hybrid synthesis.
2. Hybrid mode loads local provider config and prefers Google/Gemini when available.
3. JSON output includes observability metadata:
   - `llm_enabled`, `llm_provider`, `llm_model`, `llm_base_url`
   - `latency_ms`, `fallback_used`, `error_reason`
4. Run metadata can be exported:
   - `--save-run-metadata data/outputs/handbook_run_metadata.json`

## Verification Commands

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_handbook_writer.py tests/test_llm_client.py tests/test_cli.py -k "generate_handbook or llm_client" -q
.\.venv\Scripts\python.exe -m ruff check src/expert_digest/cli.py src/expert_digest/generation
expert-digest generate-handbook --db data/processed/zhihu_huang.sqlite3 --format json --save-run-metadata data/outputs/handbook_run_metadata.json
```

## Exit Decision

M4 is considered completed. Next milestone is M5 (Streamlit demo).
