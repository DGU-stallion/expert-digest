"""Load source articles from JSON Lines files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from expert_digest.domain.models import Document

REQUIRED_FIELDS = ("author", "title", "content", "source")


def load_jsonl_documents(path: str | Path) -> list[Document]:
    """Load documents from a UTF-8 JSONL file."""
    jsonl_path = Path(path)
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
            documents.append(
                Document.create(
                    author=str(row["author"]),
                    title=str(row["title"]),
                    content=str(row["content"]),
                    source=str(row["source"]),
                    url=_optional_string(row.get("url")),
                    created_at=_optional_string(row.get("created_at")),
                )
            )

    return documents


def _validate_required_fields(row: dict[str, Any], line_number: int) -> None:
    missing = [field for field in REQUIRED_FIELDS if not row.get(field)]
    if missing:
        fields = ", ".join(missing)
        raise ValueError(f"line {line_number}: missing required field(s): {fields}")


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
