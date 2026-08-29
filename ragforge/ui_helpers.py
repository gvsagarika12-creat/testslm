"""Pure formatting for the Streamlit layer, kept separate so it can be tested."""
from __future__ import annotations

from typing import Sequence

from ragforge.pipeline import IngestReport
from ragforge.store import Hit


def format_page_range(page_start: int, page_end: int) -> str:
    if page_start == page_end:
        return f"p{page_start}"
    return f"p{page_start}–{page_end}"


def report_rows(report: IngestReport) -> list[dict]:
    return [
        {
            "File": result.path.name,
            "Status": result.status,
            "Chunks": result.chunk_count,
            "Detail": result.message,
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
