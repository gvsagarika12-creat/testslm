"""Streamlit interface: ingest, inspect, search."""
from __future__ import annotations


import tempfile
from pathlib import Path

import streamlit as st

from ragforge.config import settings
from ragforge.llm import LLMError
from ragforge.pipeline import IngestReport, build_pipeline
from ragforge.teach import VECTOR_BADGES, VECTORS, Teacher
from ragforge.ui_helpers import format_page_range, pending_uploads, report_rows

st.set_page_config(page_title="RAGForge", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model…")
def get_pipeline():
    """Built once per session — model loading is expensive."""
    return build_pipeline()


@st.cache_resource(show_spinner=False)
def get_teacher():
    """The 3H layer. Holds no state; safe to reuse across reruns."""
    return Teacher(pipeline=get_pipeline())


pipeline = get_pipeline()

st.title("RAGForge")
st.caption("Local PDF ingestion, chunk inspection, and retrieval.")

with st.sidebar:
    st.header("Chunking")
    chunk_size = st.slider(
        "Chunk size (tokens)", 64, settings.max_model_tokens, settings.chunk_size, 8
    )
    overlap = st.slider("Overlap (tokens)", 0, 256, settings.chunk_overlap, 4)
    if overlap >= chunk_size:
        st.error("Overlap must be smaller than chunk size.")
    force = st.checkbox("Force re-ingest", value=False)

    st.divider()
    stats = pipeline.stats()
    st.metric("Documents", stats["documents"])
    st.metric("Chunks", stats["chunks"])
    st.caption(f"Vectors: {stats['backend']} · {stats['location']}")
    st.caption(f"Files: {stats['file_store']}")

ingest_tab, inspect_tab, search_tab, teach_tab = st.tabs(
    ["Ingest", "Inspect", "Search", "Teach"]
)

with ingest_tab:
    st.subheader("Upload PDFs")
    st.caption(f"Source files are kept in: `{pipeline.file_store.location}`")

    # Content hashes of uploads already put through the pipeline this session.
    # Streamlit reruns the whole script on every widget interaction and the
    # uploader keeps returning its files, so without this the same upload would
    # be re-ingested on every slider move.
    if "processed_uploads" not in st.session_state:
        st.session_state.processed_uploads = set()

    auto_ingest = st.checkbox(
        "Ingest automatically on upload",
        value=True,
        help="Chunk, embed and store each file the moment it arrives.",
    )
    uploaded = st.file_uploader(
        "Drop PDFs here", type="pdf", accept_multiple_files=True
    )

    pending = pending_uploads(
        [(item.name, bytes(item.getbuffer())) for item in (uploaded or [])],
        st.session_state.processed_uploads,
    )

    if pending:
        trigger = auto_ingest or st.button(
            f"Ingest {len(pending)} file(s)", type="primary"
        )
    else:
        trigger = False
        if uploaded:
            st.success(f"{len(uploaded)} file(s) already ingested this session.")
            if st.button("Re-ingest with current settings"):
                st.session_state.processed_uploads.clear()
                st.rerun()

    if trigger:
        results = []
        progress = st.progress(0.0, text="Starting…")
        with tempfile.TemporaryDirectory() as staging:
            for index, (key, filename, data) in enumerate(pending, start=1):
                progress.progress(
                    (index - 1) / len(pending),
                    text=f"Chunking and embedding {filename}…",
                )
                staged = Path(staging) / filename
                staged.write_bytes(data)
                results.append(
                    pipeline.ingest_file(
                        staged, chunk_size=chunk_size, overlap=overlap, force=force
                    )
                )
                # Mark done even on failure, so a broken file cannot cause an
                # endless re-ingest loop across reruns.
                st.session_state.processed_uploads.add(key)
                progress.progress(index / len(pending), text=f"{filename} done")
        st.session_state["last_report"] = report_rows(IngestReport(results=results))
        st.rerun()

    if "last_report" in st.session_state:
        rows = st.session_state["last_report"]
        failures = [r for r in rows if r["Status"] == "failed"]
        if failures:
            st.error(f"{len(failures)} file(s) failed. See Detail below.")
        st.dataframe(rows, width="stretch", hide_index=True)

    st.divider()
    st.subheader("Ingest a folder")
    folder = st.text_input("Folder path on this machine")
    recursive = st.checkbox("Include subfolders", value=True)
    if folder and st.button("Ingest folder"):
        path = Path(folder)
        if not path.is_dir():
            st.error(f"Not a directory: {folder}")
        else:
            with st.spinner("Ingesting…"):
                report = pipeline.ingest_path(
                    path,
                    recursive=recursive,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    force=force,
                )
            st.session_state["last_report"] = report_rows(report)
            st.rerun()

with inspect_tab:
    st.subheader("Inspect chunks")
    documents = pipeline.store.list_documents()
    if not documents:
        st.info("Nothing ingested yet.")
    else:
        labels = {
            f"{d.source_filename} ({d.chunk_count} chunks, "
            f"size {d.chunk_size} / overlap {d.overlap})": d.doc_id
            for d in documents
        }
        choice = st.selectbox("Document", list(labels))
        selected_doc = labels[choice]
        chunks = [c for c in pipeline.store.iter_chunks() if c.doc_id == selected_doc]
        st.caption(f"{len(chunks)} chunks")
        for chunk in chunks:
            header = (
                f"#{chunk.chunk_index} · "
                f"{format_page_range(chunk.page_start, chunk.page_end)} · "
                f"{chunk.token_count} tokens"
            )
            with st.expander(header):
                st.write(chunk.text)

with search_tab:
    st.subheader("Search")
    query = st.text_input("Query")
    k = st.slider("Results", 1, 20, 5)
    if query:
        hits = pipeline.search(query, k=k)
        if not hits:
            st.info("No results.")
        for hit in hits:
            st.markdown(
                f"**{hit.score:.3f}** · `{hit.source_filename}` · "
                f"{format_page_range(hit.page_start, hit.page_end)}"
            )
            st.write(hit.text)
            st.divider()

with teach_tab:
    st.subheader("Teach — 3H")
    st.caption(
        f"Answers are written by `{settings.ollama_model}` from retrieved passages "
        "only. Expect 60–90 seconds."
    )

    teach_question = st.text_input(
        "Learner question", key="teach_q",
        placeholder="How is central retinal artery occlusion treated?",
    )
    context_chunks = st.slider(
        "Passages to teach from", 2, 12, settings.teach_context_chunks
    )

    if st.button("Teach", type="primary", disabled=not teach_question.strip()):
        with st.spinner(f"{settings.ollama_model} is reading the passages…"):
            try:
                st.session_state["teach_answer"] = get_teacher().answer(
                    teach_question, k=context_chunks
                )
                st.session_state.pop("teach_error", None)
            except (LLMError, ValueError) as exc:
                st.session_state["teach_error"] = str(exc)
                st.session_state.pop("teach_answer", None)

    if st.session_state.get("teach_error"):
        st.error(st.session_state["teach_error"])

    answer = st.session_state.get("teach_answer")
    if answer:
        st.markdown(f"### {answer.title}")

        if answer.overview:
            st.write(answer.overview)

        for card in answer.cards:
            with st.container(border=True):
                badge, flag = st.columns([1, 6])
                with badge:
                    st.markdown(f"**{VECTOR_BADGES[card.vector]}**")
                with flag:
                    if not card.grounded:
                        st.caption("⚠️ no traceable source")
                st.markdown(f"**{card.headline}**")
                for bullet in card.bullets:
                    st.markdown(
                        f"<div style='color:#666;margin-left:1em'>– {bullet}</div>",
                        unsafe_allow_html=True,
                    )

        if answer.picture_this:
            st.markdown(f"*Picture this: {answer.picture_this}*")

        if answer.unverified_claims:
            st.warning("**[UNVERIFIED — CONFIRM WITH FACULTY]**")
            for claim in answer.unverified_claims:
                st.markdown(f"- {claim}")

        if answer.retrieval_question:
            st.info(f"**{answer.retrieval_question}**")

        if answer.sources:
            st.caption("Sources: " + " · ".join(f"`{s}`" for s in answer.sources))

        if answer.uncovered_vectors:
            st.caption(
                "Not covered by the ingested sources: "
                + ", ".join(v.upper() for v in answer.uncovered_vectors)
            )

        if answer.gap_report:
            with st.expander(f"Gap report ({len(answer.gap_report)})"):
                for gap in answer.gap_report:
                    st.markdown(f"- {gap}")

        if answer.warnings:
            with st.expander(f"⚠️ Contract warnings ({len(answer.warnings)})"):
                for warning in answer.warnings:
                    st.markdown(f"- {warning}")

        with st.expander(f"Passages this answer was written from ({len(answer.hits)})"):
            for hit in answer.hits:
                st.markdown(
                    f"**{hit.score:.3f}** · `{hit.source_filename}` · "
                    f"{format_page_range(hit.page_start, hit.page_end)}"
                )
                st.write(hit.text)
                st.divider()

        if answer.llm:
            st.caption(
                f"{answer.llm.output_tokens} tokens in "
                f"{answer.llm.duration_seconds:.1f}s "
                f"({answer.llm.tokens_per_second:.1f} tok/s)"
            )
