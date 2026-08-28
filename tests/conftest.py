"""Fixtures shared across the test suite."""
from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest


def _build_pdf(path: Path, pages: list[str], *, encrypt: bool = False) -> Path:
    """Write a small text PDF. One string per page."""
    doc = pymupdf.open()
    for body in pages:
        page = doc.new_page()
        page.insert_text((72, 72), body, fontsize=11)
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


@pytest.fixture
def tokenizer():
    return WhitespaceTokenizer()
