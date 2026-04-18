import json
from pathlib import Path

from expert_digest.cli import main
from expert_digest.domain.models import ChunkEmbedding, Handbook
from expert_digest.knowledge.topic_clusterer import (
    LLMTopicLabeler,
    TopicCluster,
    TopicRepresentativeDocument,
)


def _fake_topics() -> list[TopicCluster]:
    return [
        TopicCluster(
            topic_id="topic-1",
            label="主题1：泡泡玛特",
            chunk_count=12,
            representative_chunk_ids=["c1", "c2"],
            representative_documents=[
                TopicRepresentativeDocument(
                    document_id="doc-1",
                    title="泡泡玛特复盘",
                    author="黄彦臻",
                    url="https://example.com/p1",
                    score=0.91,
                    supporting_chunk_id="c1",
                )
            ],
        )
    ]


def _fake_embeddings():
    return [
        ChunkEmbedding.create(
            chunk_id="c1",
            model="hash-bow-v1",
            vector=[1.0, 0.0],
        )
    ]


def test_cli_cluster_topics_supports_text_output(monkeypatch, capsys):
    monkeypatch.setattr(
        "expert_digest.cli.build_topic_clusters",
        lambda **_kwargs: _fake_topics(),
    )
    monkeypatch.setattr(
        "expert_digest.cli.list_chunk_embeddings",
        lambda *_a, **_k: _fake_embeddings(),
    )

    exit_code = main(["cluster-topics"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Topic 1" in output
    assert "泡泡玛特复盘" in output


def test_cli_cluster_topics_supports_json_output(monkeypatch, capsys):
    monkeypatch.setattr(
        "expert_digest.cli.build_topic_clusters",
        lambda **_kwargs: _fake_topics(),
    )
    monkeypatch.setattr(
        "expert_digest.cli.list_chunk_embeddings",
        lambda *_a, **_k: _fake_embeddings(),
    )

    exit_code = main(["cluster-topics", "--format", "json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert len(payload["topics"]) == 1
    assert payload["topics"][0]["topic_id"] == "topic-1"
    assert (
        payload["topics"][0]["representative_documents"][0]["title"]
        == "泡泡玛特复盘"
    )


def test_cli_generate_handbook_accepts_cluster_theme_options(monkeypatch, capsys):
    captured: dict[str, object] = {}
    handbook = Handbook(
        author="黄彦臻",
        title="黄彦臻学习手册",
        markdown="# 手册",
        source_document_ids=["doc-1"],
    )

    def _fake_build_handbook(**kwargs):
        captured.update(kwargs)
        return handbook

    monkeypatch.setattr("expert_digest.cli.build_handbook", _fake_build_handbook)
    monkeypatch.setattr(
        "expert_digest.cli.write_handbook",
        lambda *, handbook, output_path: Path(output_path),
    )

    exit_code = main(
        [
            "generate-handbook",
            "--synthesis-mode",
            "deterministic",
            "--theme-source",
            "cluster",
            "--num-topics",
            "4",
        ]
    )
    _ = capsys.readouterr().out

    assert exit_code == 0
    assert captured["theme_source"] == "cluster"
    assert captured["num_topics"] == 4


def test_cli_cluster_topics_llm_mode_wires_labeler_and_metadata(monkeypatch, capsys):
    captured: dict[str, object] = {}

    class _FakeLLMClient:
        provider = "google"
        model = "gemini-2.5-flash"
        base_url = "https://generativelanguage.googleapis.com"

        def generate(self, *, system_prompt: str, user_prompt: str) -> str:
            return "价值投资与风险控制"

    def _fake_build_topic_clusters(**kwargs):
        captured.update(kwargs)
        return _fake_topics()

    monkeypatch.setattr(
        "expert_digest.cli.create_default_handbook_llm_client",
        lambda **_kwargs: _FakeLLMClient(),
    )
    monkeypatch.setattr(
        "expert_digest.cli.build_topic_clusters",
        _fake_build_topic_clusters,
    )
    monkeypatch.setattr(
        "expert_digest.cli.list_chunk_embeddings",
        lambda *_a, **_k: _fake_embeddings(),
    )

    exit_code = main(["cluster-topics", "--label-mode", "llm", "--format", "json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert isinstance(captured["labeler"], LLMTopicLabeler)
    assert payload["label_mode"] == "llm"
    assert payload["llm_provider"] == "google"


def test_cli_cluster_topics_can_save_report_payload(monkeypatch, capsys):
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "expert_digest.cli.build_topic_clusters",
        lambda **_kwargs: _fake_topics(),
    )
    monkeypatch.setattr(
        "expert_digest.cli.list_chunk_embeddings",
        lambda *_a, **_k: _fake_embeddings(),
    )

    def _fake_save_run_metadata(*, payload, output_path):
        captured["payload"] = payload
        captured["output_path"] = output_path

    monkeypatch.setattr("expert_digest.cli._save_run_metadata", _fake_save_run_metadata)

    exit_code = main(
        [
            "cluster-topics",
            "--format",
            "json",
            "--report-output",
            "data/outputs/topic_report.json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert "report" in payload
    assert payload["report"]["topic_count"] == 1
    assert captured["output_path"] == Path("data/outputs/topic_report.json")
