"""Load source articles from Zhihu crawler export files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from expert_digest.domain.models import Document

REQUIRED_FIELDS = ("source_type", "source_id", "author_name", "title")


def load_zhihu_documents(path: str | Path) -> list[Document]:
    """Load documents from Zhihu `content_index.jsonl` export."""
    jsonl_path = _resolve_content_index_path(path)
    documents: list[Document] = []

    with jsonl_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: expected a JSON object")
            _validate_required_fields(row, line_number)
            content = _extract_content(row)
            if not content:
                raise ValueError(
                    "line "
                    f"{line_number}: missing required field(s): "
                    "content_text/content_markdown/content_html"
                )
            source_type = str(row["source_type"]).strip()
            source_id = str(row["source_id"]).strip()
            documents.append(
                Document.create(
                    author=str(row["author_name"]).strip(),
                    title=str(row["title"]).strip(),
                    content=content,
                    source=f"zhihu:{source_type}:{source_id}",
                    url=_optional_string(row.get("url")),
                    created_at=_optional_string(row.get("created_at")),
                )
            )

    return documents


def _resolve_content_index_path(path: str | Path) -> Path:
    source_path = Path(path)
    if source_path.is_dir():
        candidate = source_path / "index" / "content_index.jsonl"
        if candidate.exists():
            return candidate
        fallback = source_path / "content_index.jsonl"
        if fallback.exists():
            return fallback
        raise FileNotFoundError(
            "cannot find content index under "
            f"{source_path} (expected index/content_index.jsonl)"
        )
    return source_path


def _validate_required_fields(row: dict[str, Any], line_number: int) -> None:
    missing = [
        field for field in REQUIRED_FIELDS if not str(row.get(field, "")).strip()
    ]
    if missing:
        fields = ", ".join(missing)
        raise ValueError(f"line {line_number}: missing required field(s): {fields}")


def _extract_content(row: dict[str, Any]) -> str:
    for field in ("content_text", "content_markdown", "content_html"):
        value = row.get(field)
        if value is None:
            continue
        stripped = str(value).strip()
        if stripped:
            return stripped
    return ""


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped if stripped else None
