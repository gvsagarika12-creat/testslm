"""Pure formatting for the Streamlit layer, kept separate so it can be tested."""
from __future__ import annotations

from typing import Sequence, Set

from ragforge.pipeline import IngestReport
from ragforge.store import Hit


def format_page_range(page_start: int, page_end: int) -> str:
    if page_start == page_end:
        return f"p{page_start}"
    return f"p{page_start}–{page_end}"


def upload_key(data: bytes) -> str:
    """Identity of an uploaded file: the hash of its content."""
    import hashlib

    return hashlib.sha256(data).hexdigest()


def pending_uploads(
    items: Sequence[tuple[str, bytes]], processed: Set[str]
) -> list[tuple[str, str, bytes]]:
    """Uploads not yet ingested, as (key, filename, data).

    Streamlit reruns the whole script on every widget interaction and the file
    uploader keeps returning its files, so auto-ingestion must filter against
    what it has already done or it would re-ingest on every slider move — and,
    because ingestion ends in a rerun, loop forever.

    Duplicate content within one batch collapses to a single entry: the same
    bytes are the same document regardless of filename.
    """
    seen: dict[str, tuple[str, str, bytes]] = {}
    for filename, data in items:
        key = upload_key(data)
        if key in processed or key in seen:
            continue
        seen[key] = (key, filename, data)
    return list(seen.values())


def storage_label(stored_uri: str) -> str:
    """Readable form of a stored-file URI, or "" when nothing was stored."""
    if not stored_uri:
        return ""
    if stored_uri.startswith("file:///"):
        from urllib.parse import unquote, urlparse

        return unquote(urlparse(stored_uri).path).lstrip("/")
    return stored_uri


def report_rows(report: IngestReport) -> list[dict]:
    return [
        {
            "File": result.path.name,
            "Status": result.status,
            "Chunks": result.chunk_count,
            "Detail": result.message,
            "Stored": storage_label(result.stored_uri),
        }
        for result in report.results
    ]


def hit_rows(hits: Sequence[Hit]) -> list[dict]:
    return [
        {
            "Score": round(hit.score, 3),
            "Source": hit.source_filename,
            "Pages": format_page_range(hit.page_start, hit.page_end),
            "Text": hit.text,
        }
        for hit in hits
    ]
