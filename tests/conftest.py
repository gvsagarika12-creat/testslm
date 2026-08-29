"""Fixtures shared across the test suite."""
from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest


def _build_pdf(path: Path, pages: list[str], *, encrypt: bool = False) -> Path:
    """Write a small text PDF. One string per page.

    Text is laid out with insert_textbox so it wraps. insert_text draws a single
    unwrapped line and PyMuPDF silently clips whatever runs past the page edge,
    which would quietly shorten any fixture longer than one line.
    """
    doc = pymupdf.open()
    for body in pages:
        page = doc.new_page()
        margin = 50
        box = pymupdf.Rect(
            margin, margin, page.rect.width - margin, page.rect.height - margin
        )
        overflow = page.insert_textbox(box, body, fontsize=11)
        if overflow < 0:
            raise ValueError(
                f"fixture text does not fit on one page (overflow {overflow}); "
                "split it across more pages"
            )
    if encrypt:
        doc.save(
            path,
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            owner_pw="owner",
            user_pw="user",
        )
    else:
        doc.save(path)
    doc.close()
    return path


@pytest.fixture
def make_pdf(tmp_path):
    """Factory: make_pdf(["page one text", "page two text"]) -> Path."""
    counter = {"n": 0}

    def _make(pages: list[str], *, name: str | None = None, encrypt: bool = False) -> Path:
        counter["n"] += 1
        filename = name or f"doc{counter['n']}.pdf"
        target = tmp_path / filename
        # `name` may include subdirectories, e.g. "batch/ok1.pdf".
        target.parent.mkdir(parents=True, exist_ok=True)
        return _build_pdf(target, pages, encrypt=encrypt)

    return _make


class WhitespaceTokenizer:
    """Stand-in for the real tokenizer: one token per whitespace-separated word.

    Lets chunker tests assert exact token counts without downloading a model.
    """

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def split_by_tokens(self, text: str, max_tokens: int) -> list[str]:
        words = text.split()
        return [
            " ".join(words[i : i + max_tokens])
            for i in range(0, len(words), max_tokens)
        ]

    def tail_tokens(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        return " ".join(text.split()[-max_tokens:])


@pytest.fixture
def tokenizer():
    return WhitespaceTokenizer()


# --- PostgreSQL test support ------------------------------------------------
#
# Store tests run against both backends to prove they behave identically.
# Postgres tests use a dedicated `ragforge_test` database so they can never
# touch real ingested data, and the table is truncated between tests to give
# each one the empty store the Chroma fixture gets from tmp_path.

def _admin_url() -> str:
    from ragforge.config import settings

    return settings.database_url


@pytest.fixture(scope="session")
def pg_database():
    """Factory creating isolated test databases: pg_database("name") -> url.

    Each caller gets its own database. The chunks table pins a vector width at
    creation time, so modules embedding at different dimensions must not share
    one, and no test database is ever the real corpus.
    """
    try:
        import psycopg
    except ImportError:  # pragma: no cover
        pytest.skip("psycopg not installed")

    admin = _admin_url()
    try:
        with psycopg.connect(admin, autocommit=True, connect_timeout=5) as con:
            con.execute("SELECT 1")
    except Exception as exc:
        pytest.skip(
            f"PostgreSQL not reachable ({exc.__class__.__name__}); "
            "start it with: docker compose up -d"
        )

    created: set[str] = set()

    def _make(name: str) -> str:
        if name not in created:
            with psycopg.connect(admin, autocommit=True, connect_timeout=5) as con:
                exists = con.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s", (name,)
                ).fetchone()
                if not exists:
                    con.execute(f'CREATE DATABASE "{name}"')
            created.add(name)
        return admin.rsplit("/", 1)[0] + f"/{name}"

    return _make
