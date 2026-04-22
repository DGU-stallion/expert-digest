from __future__ import annotations

import json
from pathlib import Path

from expert_digest.domain.models import Handbook
from expert_digest.generation.handbook_writer import (
    evaluate_handbook_quality,
    write_handbook,
)


def test_write_handbook_writes_trace_sidecar_with_full_paragraph_coverage(
    tmp_path: Path,
):
    handbook = Handbook(
        author="黄彦臻",
        title="黄彦臻学习手册",
        markdown=(
            "# 黄彦臻学习手册\n\n"
            "## 作者简介\n"
            "作者简介段落。\n\n"
            "## 引言\n"
            "引言段落。\n\n"
            "## 目录\n"
            "- [第一章：主题A](#第一章主题a)\n\n"
            "## 第一章：主题A\n\n"
            "### 本章目标\n"
            "理解主题A。\n\n"
            "### 核心内容\n"
            "主题A内容段落。\n\n"
            "## 结语\n"
            "总结段落。\n"
        ),
        source_document_ids=["doc-1", "doc-2"],
    )
    output_path = tmp_path / "handbook.md"

    write_handbook(handbook, output_path=output_path)

    sidecar_path = output_path.with_suffix(".trace.json")
    assert sidecar_path.exists()
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert payload["paragraph_count"] >= 3
    assert payload["covered_paragraph_count"] == payload["paragraph_count"]
    assert payload["coverage_ratio"] == 1.0
    assert payload["paragraph_traces"]


def test_evaluate_handbook_quality_detects_missing_structure_links_and_duplicates():
    markdown = (
        "# 学习手册\n\n"
        "## 引言\n"
        "重复段落。\n\n"
        "重复段落。\n\n"
        "[外链](https://example.com)\n"
    )

    report = evaluate_handbook_quality(markdown=markdown)

    assert report.structure_complete is False
    assert report.has_external_links is True
    assert report.duplicate_paragraph_ratio > 0.0
