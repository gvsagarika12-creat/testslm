from pathlib import Path

from ragforge.pipeline import FileResult, IngestReport
from ragforge.store import Hit
from ragforge.ui_helpers import (
    format_page_range,
    hit_rows,
    pending_uploads,
    report_rows,
    storage_label,
    upload_key,
)


def test_single_page_range():
    assert format_page_range(3, 3) == "p3"


def test_multi_page_range():
    assert format_page_range(3, 5) == "p3–5"


def test_report_rows_expose_status_and_message():
    report = IngestReport(results=[
        FileResult(Path("a.pdf"), "ingested", 12, "12 chunks", "file:///c:/up/a.pdf"),
        FileResult(Path("b.pdf"), "failed", 0, "password-protected"),
    ])
    rows = report_rows(report)
    assert rows[0] == {
        "File": "a.pdf",
        "Status": "ingested",
        "Chunks": 12,
        "Detail": "12 chunks",
        "Stored": "c:/up/a.pdf",
    }
    assert rows[1]["Status"] == "failed"
    assert rows[1]["Stored"] == "", "a failed file was never stored"


def test_storage_label_unescapes_a_local_uri():
    assert storage_label("file:///c:/my%20docs/a.pdf") == "c:/my docs/a.pdf"


def test_storage_label_passes_remote_uris_through():
    assert storage_label("ftp://host/docs/a.pdf") == "ftp://host/docs/a.pdf"


def test_storage_label_is_empty_when_nothing_was_stored():
    assert storage_label("") == ""


# --- auto-ingest dedupe -----------------------------------------------------


def test_all_uploads_pending_when_nothing_processed():
    items = [("a.pdf", b"one"), ("b.pdf", b"two")]
    assert [f for _, f, _ in pending_uploads(items, set())] == ["a.pdf", "b.pdf"]


def test_already_processed_uploads_are_not_returned():
    items = [("a.pdf", b"one"), ("b.pdf", b"two")]
    processed = {upload_key(b"one")}
    assert [f for _, f, _ in pending_uploads(items, processed)] == ["b.pdf"]


def test_a_second_rerun_finds_nothing_pending():
    """The loop hazard: ingestion ends in a rerun, and the uploader still holds
    the same files. If they came back pending, ingestion would never stop."""
    items = [("a.pdf", b"one"), ("b.pdf", b"two")]
    processed = set()
    first = pending_uploads(items, processed)
    processed.update(key for key, _, _ in first)
    assert pending_uploads(items, processed) == []


def test_identical_content_under_two_names_ingests_once():
    items = [("a.pdf", b"same"), ("copy-of-a.pdf", b"same")]
    assert len(pending_uploads(items, set())) == 1


def test_same_name_different_content_are_both_pending():
    items = [("a.pdf", b"one"), ("a.pdf", b"two")]
    assert len(pending_uploads(items, set())) == 2


def test_no_uploads_means_nothing_pending():
    assert pending_uploads([], set()) == []


def test_pending_uploads_carries_the_bytes_through():
    [(key, name, data)] = pending_uploads([("a.pdf", b"payload")], set())
    assert name == "a.pdf"
    assert data == b"payload"
    assert key == upload_key(b"payload")


def test_upload_key_is_content_addressed():
    assert upload_key(b"x") == upload_key(b"x")
    assert upload_key(b"x") != upload_key(b"y")


def test_hit_rows_round_scores_and_format_pages():
    hits = [Hit("c1", "d1", "some text", "report.pdf", 2, 3, 0.87654)]
    rows = hit_rows(hits)
    assert rows[0]["Score"] == 0.877
    assert rows[0]["Pages"] == "p2–3"
    assert rows[0]["Source"] == "report.pdf"
