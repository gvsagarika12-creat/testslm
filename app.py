"""Streamlit interface: ingest, inspect, search."""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from ragforge.config import settings
from ragforge.pipeline import IngestReport, build_pipeline
from ragforge.ui_helpers import format_page_range, report_rows

st.set_page_config(page_title="RAGForge", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model…")
def get_pipeline():
    """Built once per session — model loading is expensive."""
    return build_pipeline()


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

ingest_tab, inspect_tab, search_tab = st.tabs(["Ingest", "Inspect", "Search"])

with ingest_tab:
    st.subheader("Upload PDFs")
    uploaded = st.file_uploader(
        "Drop PDFs here", type="pdf", accept_multiple_files=True
    )
    if uploaded and st.button("Ingest uploads", type="primary"):
        with tempfile.TemporaryDirectory() as staging:
            results = []
            progress = st.progress(0.0)
            for index, item in enumerate(uploaded, start=1):
                staged = Path(staging) / item.name
                staged.write_bytes(item.getbuffer())
                results.append(
                    pipeline.ingest_file(
                        staged, chunk_size=chunk_size, overlap=overlap, force=force
                    )
                )
                progress.progress(index / len(uploaded))
        st.session_state["last_report"] = report_rows(IngestReport(results=results))
        st.rerun()

    if "last_report" in st.session_state:
        st.dataframe(
            st.session_state["last_report"],
            width="stretch",
            hide_index=True,
        )

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
