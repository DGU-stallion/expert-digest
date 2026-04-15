"""Load source articles from Markdown files."""

from __future__ import annotations

from pathlib import Path

from expert_digest.domain.models import Document


def load_markdown_documents(folder: str | Path) -> list[Document]:
    """Load every Markdown document under a folder in stable path order."""
    root = Path(folder)
    documents: list[Document] = []

    for markdown_path in sorted(root.rglob("*.md")):
        raw_text = markdown_path.read_text(encoding="utf-8")
        metadata, content = _split_front_matter(raw_text)
        stripped_content = content.strip()
        title = (
            metadata.get("title")
            or _extract_h1(stripped_content)
            or markdown_path.stem
        )
        author = metadata.get("author") or "unknown"
        documents.append(
            Document.create(
                author=author,
                title=title,
                content=stripped_content,
                source=str(markdown_path),
                url=metadata.get("url"),
                created_at=metadata.get("created_at"),
            )
        )

    return documents


def _split_front_matter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    metadata: dict[str, str] = {}
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            content = "\n".join(lines[index + 1 :])
            return metadata, content
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()

    return {}, text


def _extract_h1(content: str) -> str | None:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped.removeprefix("# ").strip()
    return None
