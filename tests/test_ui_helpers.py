from pathlib import Path

from ragforge.pipeline import FileResult, IngestReport
from ragforge.store import Hit
from ragforge.ui_helpers import format_page_range, hit_rows, report_rows


def test_single_page_range():
    assert format_page_range(3, 3) == "p3"


def test_multi_page_range():
    assert format_page_range(3, 5) == "p3–5"


def test_report_rows_expose_status_and_message():
    report = IngestReport(results=[
        FileResult(Path("a.pdf"), "ingested", 12, "12 chunks"),
        FileResult(Path("b.pdf"), "failed", 0, "password-protected"),
    ])
    rows = report_rows(report)
    assert rows[0] == {"File": "a.pdf", "Status": "ingested", "Chunks": 12,
                       "Detail": "12 chunks"}
    assert rows[1]["Status"] == "failed"


def test_hit_rows_round_scores_and_format_pages():
    hits = [Hit("c1", "d1", "some text", "report.pdf", 2, 3, 0.87654)]
    rows = hit_rows(hits)
    assert rows[0]["Score"] == 0.877
    assert rows[0]["Pages"] == "p2–3"
    assert rows[0]["Source"] == "report.pdf"
