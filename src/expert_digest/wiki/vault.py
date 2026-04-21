"""Markdown vault filesystem operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from expert_digest.wiki.frontmatter import parse_frontmatter, render_frontmatter
from expert_digest.wiki.models import WikiPage


@dataclass(frozen=True)
class WikiVault:
    root: Path

    def initialize(
        self,
        *,
        expert_id: str,
        expert_name: str,
        purpose: str,
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for folder in ("sources", "concepts", "topics", "theses", "reviews"):
            (self.root / folder).mkdir(parents=True, exist_ok=True)

        self._write_if_missing(
            "purpose.md",
            f"# Purpose\n\n专家：{expert_name}\n\n{purpose}\n",
        )
        self._write_if_missing(
            "schema.md",
            "# Schema\n\n"
            "页面类型：source、concept、topic、thesis、review。\n\n"
            "核心判断必须包含 source refs。\n",
        )
        self._write_if_missing(
            "index.md",
            f"# {expert_name} Expert Wiki\n\n"
            f"- Expert ID: `{expert_id}`\n\n"
            "## Sources\n\n## Topics\n\n## Concepts\n",
        )
        self._write_if_missing(
            "log.md",
            "# Log\n\n",
        )

    def write_page(self, page: WikiPage) -> Path:
        path = self.root / page.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_frontmatter(page), encoding="utf-8")
        return path

    def read_page(self, relative_path: str | Path) -> WikiPage:
        path = self.root / relative_path
        text = path.read_text(encoding="utf-8")
        return parse_frontmatter(
            text,
            path=Path(relative_path).as_posix(),
        )

    def list_pages(self) -> list[WikiPage]:
        pages: list[WikiPage] = []
        for path in sorted(self.root.rglob("*.md")):
            relative = path.relative_to(self.root).as_posix()
            pages.append(self.read_page(relative))
        return pages

    def append_log(self, line: str) -> None:
        path = self.root / "log.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(line.rstrip() + "\n")

    def _write_if_missing(self, relative_path: str, content: str) -> None:
        path = self.root / relative_path
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
