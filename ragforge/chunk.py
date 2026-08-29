"""Split page text into token-bounded chunks that keep page provenance.

Two stages. Segmentation breaks pages down along natural boundaries until every
segment fits the token budget: paragraph, then sentence, then word group, then a
hard token cut as a last resort. Packing greedily fills chunks with whole
segments and carries the trailing segments of each chunk into the next as
overlap.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from ragforge.parse import PageText


class Tokenizer(Protocol):
    def count_tokens(self, text: str) -> int: ...

    def split_by_tokens(self, text: str, max_tokens: int) -> list[str]: ...

    def tail_tokens(self, text: str, max_tokens: int) -> str: ...


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    text: str
    source_filename: str
    page_start: int
    page_end: int
    chunk_index: int
    token_count: int
    chunk_size: int
    overlap: int
    ingested_at: str


@dataclass(frozen=True)
class _Segment:
    text: str
    page_number: int
    token_count: int


_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def chunk_id(doc_id: str, chunk_index: int) -> str:
    return hashlib.sha256(f"{doc_id}:{chunk_index}".encode()).hexdigest()[:32]


def _split_words(text: str, tokenizer: Tokenizer, budget: int) -> list[str]:
    """Group words into runs that fit the budget, without breaking words."""
    pieces: list[str] = []
    current: list[str] = []
    for word in text.split():
        candidate = " ".join(current + [word])
        if current and tokenizer.count_tokens(candidate) > budget:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _segment_page(
    page: PageText, tokenizer: Tokenizer, budget: int, stride: int
) -> list[_Segment]:
    """Break one page into segments that each fit the budget.

    Natural units (paragraphs, sentences) are kept whole as long as they fit
    `budget`. Anything that must be cut artificially is cut to `stride`
    (budget minus overlap) instead, so that a carried overlap tail plus a
    following segment still fits inside one chunk. Cutting to `budget` here
    would make overlap impossible for any text dense enough to need splitting.
    """
    segments: list[_Segment] = []

    def emit(text: str) -> None:
        text = text.strip()
        if text:
            segments.append(
                _Segment(
                    text=text,
                    page_number=page.page_number,
                    token_count=tokenizer.count_tokens(text),
                )
            )

    for paragraph in _PARAGRAPH.split(page.text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if tokenizer.count_tokens(paragraph) <= budget:
            emit(paragraph)
            continue
        for sentence in _SENTENCE_END.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            if tokenizer.count_tokens(sentence) <= budget:
                emit(sentence)
                continue
            for run in _split_words(sentence, tokenizer, stride):
                if tokenizer.count_tokens(run) <= stride:
                    emit(run)
                else:
                    # A single token-dense word: only a hard cut can bound it.
                    for piece in tokenizer.split_by_tokens(run, stride):
                        emit(piece)
    return segments


def _overlap_tail(
    segments: Sequence[_Segment], overlap: int, tokenizer: Tokenizer
) -> list[_Segment]:
    """Trailing context to carry into the next chunk, at most `overlap` tokens."""
    if overlap <= 0 or not segments:
        return []

    tail: list[_Segment] = []
    total = 0
    for segment in reversed(segments):
        if total + segment.token_count > overlap:
            break
        tail.insert(0, segment)
        total += segment.token_count
    if tail:
        return tail

    # No whole segment fits the overlap budget. Carry the trailing tokens of the
    # last segment instead, so consecutive chunks always share context.
    last = segments[-1]
    text = tokenizer.tail_tokens(last.text, overlap).strip()
    if not text:
        return []
    return [
        _Segment(
            text=text,
            page_number=last.page_number,
            token_count=tokenizer.count_tokens(text),
        )
    ]


def chunk_pages(
    pages: Sequence[PageText],
    *,
    doc_id: str,
    source_filename: str,
    tokenizer: Tokenizer,
    chunk_size: int,
    overlap: int,
    ingested_at: str,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    stride = chunk_size - overlap
    segments: list[_Segment] = []
    for page in pages:
        segments.extend(_segment_page(page, tokenizer, chunk_size, stride))
    if not segments:
        return []

    chunks: list[Chunk] = []
    current: list[_Segment] = []
    current_tokens = 0

    def emit_chunk(members: list[_Segment]) -> None:
        text = " ".join(s.text for s in members)
        index = len(chunks)
        chunks.append(
            Chunk(
                id=chunk_id(doc_id, index),
                doc_id=doc_id,
                text=text,
                source_filename=source_filename,
                page_start=min(s.page_number for s in members),
                page_end=max(s.page_number for s in members),
                chunk_index=index,
                token_count=tokenizer.count_tokens(text),
                chunk_size=chunk_size,
                overlap=overlap,
                ingested_at=ingested_at,
            )
        )

    for segment in segments:
        if current and current_tokens + segment.token_count > chunk_size:
            emit_chunk(current)
            carried = _overlap_tail(current, overlap, tokenizer)
            current = list(carried)
            current_tokens = sum(s.token_count for s in carried)
            # The carried tail plus this segment may still overflow; drop the
            # tail rather than exceed the budget.
            if current_tokens + segment.token_count > chunk_size:
                current, current_tokens = [], 0
        current.append(segment)
        current_tokens += segment.token_count

    # Final flush. No overlap tail is carried past the last chunk.
    if current:
        emit_chunk(current)

    return chunks
