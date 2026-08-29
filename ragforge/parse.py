"""Extract normalized, page-attributed text from digital PDFs."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf


class PdfParseError(Exception):
    """Base class for every failure that should skip one file, not the batch."""


class EncryptedPdfError(PdfParseError):
    pass


class NoTextLayerError(PdfParseError):
    pass


class CorruptPdfError(PdfParseError):
    pass


@dataclass(frozen=True)
class PageText:
    page_number: int  # 1-indexed
    text: str


_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_BLANK_RUN = re.compile(r"\n{3,}")
_LONE_NEWLINE = re.compile(r"(?<!\n)\n(?!\n)")
_HORIZONTAL_SPACE = re.compile(r"[ \t]{2,}")
_SPACE_AROUND_NEWLINE = re.compile(r"[ \t]*\n[ \t]*")


def normalize_text(raw: str) -> str:
    """Clean extracted text while preserving paragraph structure.

    Paragraph breaks are load-bearing: the chunker splits on them first.
    """
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _BLANK_RUN.sub("\n\n", text)
    text = _LONE_NEWLINE.sub(" ", text)
    text = _HORIZONTAL_SPACE.sub(" ", text)
    text = _SPACE_AROUND_NEWLINE.sub("\n", text)
    return text.strip()


def compute_doc_id(data: bytes) -> str:
    """Content hash of a file. Identical bytes always yield the same id."""
    return hashlib.sha256(data).hexdigest()


def extract_pages(pdf_path: Path) -> list[PageText]:
    """Return normalized text per page, dropping pages that are empty.

    Raises a PdfParseError subclass so callers can skip one file and continue.
    """
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as exc:  # PyMuPDF raises assorted types for bad input
        raise CorruptPdfError(f"could not open PDF: {exc}") from exc

    try:
        if doc.needs_pass:
            raise EncryptedPdfError("PDF is password-protected")

        pages: list[PageText] = []
        for index, page in enumerate(doc, start=1):
            try:
                raw = page.get_text("text")
            except Exception as exc:
                raise CorruptPdfError(f"could not read page {index}: {exc}") from exc
            cleaned = normalize_text(raw)
            if cleaned:
                pages.append(PageText(page_number=index, text=cleaned))
    finally:
        doc.close()

    if not pages:
        raise NoTextLayerError(
            "no text layer — likely a scanned document, OCR not enabled"
        )
    return pages
