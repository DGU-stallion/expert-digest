"""Streamlit demo app for ExpertDigest M5."""

from __future__ import annotations

from pathlib import Path

from expert_digest.app import services
from expert_digest.generation.llm_client import DEFAULT_CCSWITCH_DB_PATH
from expert_digest.processing.embedder import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EMBEDDING_MODEL,
)
from expert_digest.storage.sqlite_store import DEFAULT_DATABASE_PATH


def _load_streamlit():
    try:
        import streamlit as st
    except ImportError as error:  # pragma: no cover
        raise RuntimeError(
            "Streamlit is not installed. Install with `pip install -e \".[app]\"`."
        ) from error
    return st


def run() -> None:
    st = _load_streamlit()
    st.set_page_config(page_title="ExpertDigest M5 Demo", layout="wide")
    st.title("ExpertDigest M5 - Streamlit Demo")
    st.caption("基础闭环：导入 -> 处理 -> 问答 -> 手册预览")

    db_path_raw = st.sidebar.text_input("数据库路径", value=str(DEFAULT_DATABASE_PATH))
    model = st.sidebar.text_input("Embedding 模型", value=DEFAULT_EMBEDDING_MODEL)
    page = st.sidebar.radio(
        "页面",
        ("导入数据", "处理数据", "问答检索", "手册预览"),
    )
    db_path = Path(db_path_raw.strip() or DEFAULT_DATABASE_PATH)

    if page == "导入数据":
        _render_import_page(st=st, db_path=db_path)
    elif page == "处理数据":
        _render_process_page(st=st, db_path=db_path, model=model)
    elif page == "问答检索":
        _render_ask_page(st=st, db_path=db_path, model=model)
    else:
        _render_handbook_page(st=st, db_path=db_path, model=model)


def _render_import_page(*, st, db_path: Path) -> None:
    st.subheader("导入数据")
    kind = st.selectbox("导入类型", ["jsonl", "markdown", "zhihu"], index=0)
    default_source = {
        "jsonl": "data/sample/articles.jsonl",
        "markdown": "data/sample/markdown",
        "zhihu": "D:/Project/Zhihu_Crawler/data/zhihu/huang-wei-yan-30",
    }[kind]
    uploaded_jsonl = None
    source_label = "来源路径"
    if kind == "jsonl":
        uploaded_jsonl = st.file_uploader(
            "上传 JSONL 文件",
            type=["jsonl"],
            accept_multiple_files=False,
        )
        st.caption("支持直接上传 JSONL。若同时填写路径，将优先使用上传文件。")
        source_label = "本地 JSONL 路径（可选）"

    source_path = st.text_input(source_label, value=default_source)

    if st.button("执行导入", type="primary", use_container_width=True):
        if kind == "jsonl" and uploaded_jsonl is not None:
            resolved_source_path = services.persist_uploaded_jsonl(
                filename=uploaded_jsonl.name,
                content=uploaded_jsonl.getvalue(),
            )
        elif source_path.strip():
            resolved_source_path = Path(source_path.strip())
        else:
            st.warning("请上传文件或填写来源路径。")
            return

        try:
            count = services.import_documents(
                kind=kind,
                source_path=resolved_source_path,
                db_path=db_path,
            )
        except Exception as error:  # pragma: no cover
            st.error(f"导入失败: {error}")
            return
        st.success(f"导入完成：{count} 篇文档写入 {db_path}")


def _render_process_page(*, st, db_path: Path, model: str) -> None:
    st.subheader("处理数据")
    try:
        overview = services.collect_data_overview(db_path=db_path, model=model)
    except Exception as error:  # pragma: no cover
        st.error(f"读取数据概览失败: {error}")
        return

    metric_columns = st.columns(4)
    metric_columns[0].metric("文档数", overview.document_count)
    metric_columns[1].metric("Chunk 数", overview.chunk_count)
    metric_columns[2].metric("Embedding 数", overview.embedding_count)
    metric_columns[3].metric("作者数", len(overview.authors))
    author_list = "、".join(overview.authors) if overview.authors else "(无)"
    st.caption("作者列表: " + author_list)

    with st.form("rebuild_chunks_form"):
        st.markdown("### 重建 Chunks")
        max_chars = st.number_input("max_chars", min_value=100, value=1200, step=100)
        min_chars = st.number_input("min_chars", min_value=1, value=80, step=1)
        submitted = st.form_submit_button(
            "执行 rebuild-chunks",
            use_container_width=True,
        )
        if submitted:
            try:
                count = services.rebuild_chunks(
                    db_path=db_path,
                    max_chars=int(max_chars),
                    min_chars=int(min_chars),
                )
            except Exception as error:  # pragma: no cover
                st.error(f"重建 chunks 失败: {error}")
            else:
                st.success(f"rebuild-chunks 完成：{count} 条 chunk")

    with st.form("rebuild_embeddings_form"):
        st.markdown("### 重建 Embeddings")
        dim = st.number_input(
            "embedding_dim",
            min_value=8,
            value=DEFAULT_EMBEDDING_DIM,
            step=8,
        )
        submitted = st.form_submit_button(
            "执行 rebuild-embeddings",
            use_container_width=True,
        )
        if submitted:
            try:
                count = services.rebuild_embeddings(
                    db_path=db_path,
                    model=model,
                    dim=int(dim),
                )
            except Exception as error:  # pragma: no cover
                st.error(f"重建 embeddings 失败: {error}")
            else:
                st.success(f"rebuild-embeddings 完成：{count} 条 embedding")

    _render_topic_cluster_block(st=st, db_path=db_path, model=model)


def _render_topic_cluster_block(*, st, db_path: Path, model: str) -> None:
    st.markdown("---")
    st.subheader("M6 主题聚类概览")
    num_topics = st.slider(
        "num_topics",
        min_value=1,
        max_value=10,
        value=3,
        key="cluster_num_topics",
    )
    top_docs = st.slider(
        "top_docs_per_topic",
        min_value=1,
        max_value=5,
        value=2,
        key="cluster_top_docs",
    )
    max_iter = st.slider(
        "max_iter",
        min_value=10,
        max_value=100,
        value=30,
        key="cluster_max_iter",
    )
    label_mode = st.selectbox(
        "topic 标签模式",
        options=["deterministic", "llm"],
        index=0,
        key="cluster_label_mode",
    )
    report_output = st.text_input(
        "聚类报告输出路径（可选）",
        value="data/outputs/topic_report.json",
        key="cluster_report_output",
    )
    with st.expander("LLM 命名参数（可选）", expanded=False):
        ccswitch_db = st.text_input(
            "ccswitch_db_path (cluster)",
            value=str(DEFAULT_CCSWITCH_DB_PATH),
            key="cluster_ccswitch_db",
        )
        llm_timeout = st.number_input(
            "llm_timeout (cluster)",
            min_value=5,
            max_value=60,
            value=20,
            step=1,
            key="cluster_llm_timeout",
        )

    if st.button("生成主题聚类报告", type="primary", use_container_width=True):
        try:
            result = services.cluster_topics(
                db_path=db_path,
                model=model,
                num_topics=num_topics,
                top_docs=top_docs,
                max_iter=max_iter,
                label_mode=label_mode,
                ccswitch_db_path=Path(ccswitch_db),
                llm_timeout=int(llm_timeout),
                report_output=Path(report_output.strip())
                if report_output.strip()
                else None,
            )
        except Exception as error:  # pragma: no cover
            st.error(f"主题聚类失败: {error}")
            return

        report = result.report
        metric_columns = st.columns(4)
        metric_columns[0].metric("topic_count", report.topic_count)
        metric_columns[1].metric("total_chunks", report.total_chunks)
        metric_columns[2].metric(
            "largest_topic_ratio",
            f"{report.largest_topic_ratio:.4f}",
        )
        mean_intra = (
            f"{report.mean_intra_similarity_proxy:.4f}"
            if report.mean_intra_similarity_proxy is not None
            else "(无)"
        )
        metric_columns[3].metric("mean_intra_proxy", mean_intra)

        if result.topics:
            st.markdown("#### 主题分布")
            st.bar_chart({topic.label: topic.chunk_count for topic in result.topics})

            st.markdown("#### 主题摘要")
            st.table(
                [
                    {
                        "topic": item.label,
                        "chunks": item.chunk_count,
                        "rep_docs": item.representative_document_count,
                        "mean_rep_score": (
                            f"{item.mean_representative_score:.4f}"
                            if item.mean_representative_score is not None
                            else "(无)"
                        ),
                    }
                    for item in report.topics
                ]
            )

            representative_rows: list[dict[str, str]] = []
            for topic in result.topics:
                for document in topic.representative_documents:
                    representative_rows.append(
                        {
                            "topic": topic.label,
                            "title": document.title,
                            "author": document.author,
                            "score": f"{document.score:.4f}",
                        }
                    )
            st.markdown("#### 代表原文")
            if representative_rows:
                st.table(representative_rows)
            else:
                st.write("(无)")
        else:
            st.info("当前没有可用主题（请先确认 embeddings 是否已构建）。")

        if result.report_output is not None:
            st.success(f"聚类报告已导出：{result.report_output}")


def _render_ask_page(*, st, db_path: Path, model: str) -> None:
    st.subheader("问答检索")
    question = st.text_area(
        "输入问题",
        value="泡泡玛特的核心能力是什么？",
        height=100,
    )
    top_k = st.slider("top_k", min_value=1, max_value=10, value=3)
    max_evidence = st.slider("max_evidence", min_value=1, max_value=10, value=3)
    min_top_score = st.slider(
        "min_top_score",
        min_value=0.0,
        max_value=1.0,
        value=0.05,
    )
    min_avg_score = st.slider(
        "min_avg_score",
        min_value=0.0,
        max_value=1.0,
        value=0.03,
    )

    if st.button("执行问答", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("请先输入问题。")
            return
        try:
            result = services.ask(
                question=question.strip(),
                db_path=db_path,
                model=model,
                top_k=top_k,
                max_evidence=max_evidence,
                min_top_score=min_top_score,
                min_avg_score=min_avg_score,
            )
        except Exception as error:  # pragma: no cover
            st.error(f"问答失败: {error}")
            return

        st.markdown("### 回答")
        st.write(result.answer)
        st.markdown("### 不确定性")
        st.write(result.uncertainty)
        if result.refused:
            st.warning("当前已触发拒答策略（证据不足或置信度不达标）。")
        else:
            st.success("已生成结构化回答。")

        st.markdown("### 依据")
        if not result.evidence:
            st.write("(无)")
        else:
            st.table(
                [
                    {
                        "score": f"{item.score:.4f}",
                        "title": item.title,
                        "author": item.author,
                        "snippet": item.snippet,
                    }
                    for item in result.evidence
                ]
            )

        st.markdown("### 推荐原文")
        if not result.recommended_original:
            st.write("(无)")
        else:
            for index, item in enumerate(result.recommended_original, start=1):
                st.write(f"{index}. {item}")


def _render_handbook_page(*, st, db_path: Path, model: str) -> None:
    st.subheader("手册预览")
    author = st.text_input("作者过滤（可选）", value="")
    theme_source = st.selectbox(
        "主题组织方式",
        options=["preset", "cluster"],
        index=0,
    )
    if theme_source == "cluster":
        num_topics = st.slider("num_topics", min_value=1, max_value=10, value=3)
    else:
        num_topics = 3
    top_k = st.slider("theme top_k", min_value=1, max_value=10, value=3)
    max_themes = st.slider("max_themes", min_value=1, max_value=8, value=3)
    synthesis_mode = st.selectbox(
        "synthesis_mode",
        options=["hybrid", "deterministic"],
        index=0,
    )
    output_path = st.text_input(
        "输出路径",
        value="data/outputs/handbook_from_streamlit.md",
    )

    with st.expander("LLM 参数（为后续 API 接入预留）", expanded=False):
        ccswitch_db = st.text_input(
            "ccswitch_db_path",
            value=str(DEFAULT_CCSWITCH_DB_PATH),
        )
        llm_timeout = st.number_input("llm_timeout", min_value=5, value=30, step=1)
        llm_max_tokens = st.number_input(
            "llm_max_tokens",
            min_value=128,
            value=700,
            step=64,
        )

    if st.button("生成手册", type="primary", use_container_width=True):
        try:
            result = services.generate_handbook(
                db_path=db_path,
                author=author.strip() or None,
                model=model,
                top_k=top_k,
                max_themes=max_themes,
                output_path=Path(output_path),
                synthesis_mode=synthesis_mode,
                theme_source=theme_source,
                num_topics=num_topics,
                ccswitch_db_path=Path(ccswitch_db),
                llm_timeout=int(llm_timeout),
                llm_max_tokens=int(llm_max_tokens),
            )
        except Exception as error:  # pragma: no cover
            st.error(f"生成手册失败: {error}")
            return

        st.success(f"手册已生成：{result.output_path}")
        st.markdown("### 预览")
        st.markdown(result.handbook.markdown)


if __name__ == "__main__":
    run()
