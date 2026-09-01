"""Streamlit interface: ingest, inspect, search."""
from __future__ import annotations


import tempfile
from pathlib import Path

import streamlit as st

from ragforge.config import settings
from ragforge.llm import LLMError
from ragforge.pipeline import IngestReport, build_pipeline
from ragforge.simulate import Simulator
from ragforge.teach import VECTOR_BADGES, VECTORS, Teacher
from ragforge.ui_helpers import format_page_range, pending_uploads, report_rows

st.set_page_config(page_title="RAGForge", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model…")
def get_pipeline():
    """Built once per session — model loading is expensive."""
    return build_pipeline()


def get_teacher():
    """The 3H layer.

    Deliberately not cached. It only holds references — the expensive part, the
    embedding model, lives in the cached pipeline. Caching it would pin a stale
    instance across code changes, which with the file watcher disabled means an
    edited method appears missing until the whole app is restarted.
    """
    return Teacher(pipeline=get_pipeline())


def get_simulator():
    """The case simulator. Not cached, for the same reason as get_teacher."""
    return Simulator(pipeline=get_pipeline())


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
    mode = st.radio(
        "Mode",
        ["Explain a topic", "Run a case"],
        horizontal=True,
        label_visibility="collapsed",
        help="Explaining teaches from the passages. A case puts a patient in front of you, and is the only mode that can score HEART.",
    )

    if mode == "Explain a topic":
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
                    # A new question means the previous grading no longer applies.
                    st.session_state.pop("assessment", None)
                    st.session_state.pop("assess_error", None)
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

                learner_answer = st.text_area(
                    "Your answer", key="learner_answer", height=100,
                    placeholder="Answer in your own words — then submit for grading.",
                )
                if st.button(
                    "Submit answer", disabled=not learner_answer.strip(), key="submit_answer"
                ):
                    with st.spinner("Grading your answer…"):
                        try:
                            st.session_state["assessment"] = get_teacher().assess(
                                answer.retrieval_question, learner_answer, answer.hits
                            )
                            st.session_state.pop("assess_error", None)
                        except (LLMError, ValueError) as exc:
                            st.session_state["assess_error"] = str(exc)
                            st.session_state.pop("assessment", None)

                if st.session_state.get("assess_error"):
                    st.error(st.session_state["assess_error"])

                assessment = st.session_state.get("assessment")
                if assessment:
                    with st.container(border=True):
                        verdict_style = {
                            "correct": st.success,
                            "partially_correct": st.warning,
                            "incorrect": st.error,
                        }.get(assessment.verdict, st.info)
                        verdict_style(f"**{assessment.verdict_label}**")

                        if assessment.acknowledgement:
                            st.write(assessment.acknowledgement)

                        if assessment.assessed_scores:
                            columns = st.columns(len(assessment.assessed_scores))
                            for column, score in zip(columns, assessment.assessed_scores):
                                column.metric(score.badge, f"{score.level}/4", score.anchor)

                        if assessment.what_was_right:
                            st.markdown("**What you got right**")
                            for point in assessment.what_was_right:
                                st.markdown(f"- {point}")

                        if assessment.what_was_missed:
                            st.markdown("**What was missing**")
                            for point in assessment.what_was_missed:
                                st.markdown(f"- {point}")

                        if assessment.model_answer:
                            st.markdown("**The answer**")
                            st.write(assessment.model_answer)
                            if assessment.citations:
                                st.caption(
                                    "Sources: "
                                    + " · ".join(f"`{c}`" for c in assessment.citations)
                                )

                        if assessment.feed_forward:
                            st.markdown(f"**Next:** {assessment.feed_forward}")

                        if assessment.needs_faculty_review:
                            st.warning(
                                "🚩 **FACULTY REVIEW** — the grader was not confident "
                                "in this assessment."
                            )

                        if assessment.warnings:
                            with st.expander("⚠️ Grading warnings"):
                                for warning in assessment.warnings:
                                    st.markdown(f"- {warning}")

                        if assessment.llm:
                            st.caption(
                                f"graded in {assessment.llm.duration_seconds:.0f}s · "
                                f"confidence {assessment.grader_confidence}"
                            )

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


    else:
        st.caption(
            "A three-scene case: reasoning, then the patient's distress, then "
            "management. HEART is scored from what you actually say to the patient — "
            "which a factual question can never measure."
        )

        case = st.session_state.get("case")

        if case is None:
            topic = st.text_input(
                "Case topic", key="case_topic",
                placeholder="central retinal artery occlusion",
            )
            if st.button("Start case", type="primary", disabled=not topic.strip()):
                with st.spinner("Building the case…"):
                    try:
                        st.session_state["case"] = get_simulator().start(topic)
                        st.session_state.pop("case_error", None)
                        st.rerun()
                    except (LLMError, ValueError) as exc:
                        st.session_state["case_error"] = str(exc)
            if st.session_state.get("case_error"):
                st.error(st.session_state["case_error"])

        else:
            header, control = st.columns([5, 1])
            header.markdown(f"### {case.title}")
            if control.button("End case"):
                for key in ("case", "case_error"):
                    st.session_state.pop(key, None)
                st.rerun()

            st.write(case.presentation)
            st.markdown(f"*The patient: {case.persona}*")
            if case.citations:
                st.caption("Sources: " + " · ".join(f"`{c}`" for c in case.citations))

            for turn in case.turns:
                with st.container(border=True):
                    st.markdown(f"**{VECTOR_BADGES[turn.scene.vector]} · scene "
                                f"{turn.scene.index + 1}**")
                    st.write(turn.scene.situation)
                    st.caption(turn.scene.prompt)
                    st.markdown(f"> {turn.learner_reply}")
                    if turn.reaction:
                        st.write(turn.reaction)

            if not case.finished:
                scene = case.scene
                with st.container(border=True):
                    st.markdown(f"**{VECTOR_BADGES[scene.vector]} · scene "
                                f"{scene.index + 1} of 3**")
                    st.write(scene.situation)
                    st.markdown(f"**{scene.prompt}**")
                    reply = st.text_area(
                        "Your response", key=f"scene_{scene.index}", height=110
                    )
                    if st.button("Respond", type="primary", disabled=not reply.strip()):
                        with st.spinner("…"):
                            try:
                                st.session_state["case"] = get_simulator().respond(case, reply)
                                st.rerun()
                            except (LLMError, ValueError) as exc:
                                st.error(str(exc))

            elif case.debrief is None:
                st.success("All three scenes complete.")
                if st.button("Score this case", type="primary"):
                    with st.spinner("Scoring against the §8 anchors…"):
                        try:
                            get_simulator().score(case)
                            st.session_state["case"] = case
                            st.rerun()
                        except (LLMError, ValueError) as exc:
                            st.error(str(exc))

            if case.debrief:
                d = case.debrief
                st.markdown("### Debrief")
                with st.container(border=True):
                    {"correct": st.success, "partially_correct": st.warning,
                     "incorrect": st.error}.get(d.verdict, st.info)(f"**{d.verdict_label}**")

                    if d.acknowledgement:
                        st.write(d.acknowledgement)

                    columns = st.columns(3)
                    for column, s in zip(columns, d.scores):
                        column.metric(s.badge, f"{s.level}/4", s.anchor)
                    for s in d.scores:
                        if s.evidence:
                            st.caption(f"**{s.badge}** — your words: “{s.evidence}”")

                    if d.what_was_right:
                        st.markdown("**What you did well**")
                        for point in d.what_was_right:
                            st.markdown(f"- {point}")
                    if d.what_was_missed:
                        st.markdown("**What was missing**")
                        for point in d.what_was_missed:
                            st.markdown(f"- {point}")
                    if d.model_answer:
                        st.markdown("**How a strong learner handles this case**")
                        st.write(d.model_answer)
                        if d.citations:
                            st.caption("Sources: " + " · ".join(f"`{c}`" for c in d.citations))
                    if d.feed_forward:
                        st.markdown(f"**Next:** {d.feed_forward}")
                    if d.needs_faculty_review:
                        st.warning("🚩 **FACULTY REVIEW** — the grader was not confident.")
                    if d.warnings:
                        with st.expander("⚠️ Grading warnings"):
                            for warning in d.warnings:
                                st.markdown(f"- {warning}")

            if case.warnings:
                with st.expander(f"⚠️ Case warnings ({len(case.warnings)})"):
                    for warning in case.warnings:
                        st.markdown(f"- {warning}")
