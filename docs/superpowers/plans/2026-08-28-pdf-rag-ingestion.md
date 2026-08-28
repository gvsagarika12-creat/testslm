# PDF RAG Ingestion Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local pipeline that turns uploaded PDFs into token-aware chunks stored in a vector database, with a Streamlit interface for ingestion/inspection/search and a CLI for batch ingestion.

**Architecture:** A pure core library (`parse` → `chunk` → `embed` → `store`) orchestrated by `pipeline.py`, consumed by two thin entrypoints (`app.py` Streamlit, `cli.py` Typer). Parsing, chunking, and tokenization never touch the database, so they unit-test with fakes and no model download. The vector store sits behind a Protocol so ChromaDB can be replaced without editing callers.

**Tech Stack:** Python 3.11+, PyMuPDF, sentence-transformers (`BAAI/bge-small-en-v1.5`), ChromaDB (embedded), Streamlit, Typer, pydantic-settings, pytest.

**Spec:** `docs/superpowers/specs/2026-08-28-pdf-rag-ingestion-design.md`

## Global Constraints

- Python 3.11 or newer. Python is **not currently installed** on this machine; Task 1 installs it.
- Everything runs locally. No cloud APIs, no network calls at query time, no API keys anywhere.
- Embedding runs on **CPU**. The machine's GPU (Quadro P600, 2 GB) is not used and must not be required.
- Embedding model: `BAAI/bge-small-en-v1.5`, 384 dimensions.
- `max_model_tokens = 510` (the model's 512-token window minus the two special tokens the encoder adds). **No chunk may ever exceed this.** Chunk text and its vector must represent the same content.
- Default chunk parameters: `chunk_size = 400` tokens, `chunk_overlap = 60` tokens. Both must be overridable per ingest from the UI and the CLI.
- `doc_id = sha256(file bytes)`; `chunk.id = sha256(doc_id + ":" + chunk_index)[:32]`. Both deterministic across runs.
- Unit tests must run with **no network access and no model download**. Only tests marked `slow` may download the model.
- Out of scope, do not build: OCR, LLM generation, reranking, auth, table extraction, job queues.
- Vector similarity is **cosine**. Chroma collections must be created with `metadata={"hnsw:space": "cosine"}`.
- All file paths below are relative to the repo root `c:\Users\admin\Desktop\testslm`.

---

### Task 1: Environment, scaffolding, and configuration

Installs Python, creates the package skeleton, and delivers the typed settings object every later task imports.

**Files:**
- Create: `requirements.txt`
- Create: `pytest.ini`
- Create: `ragforge/__init__.py`
- Create: `ragforge/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ragforge.config.Settings` (pydantic-settings model) and the module-level singleton `settings`. Fields: `embedding_model_name: str`, `max_model_tokens: int`, `chunk_size: int`, `chunk_overlap: int`, `embed_batch_size: int`, `collection_name: str`, `data_dir: Path`. Properties: `uploads_dir: Path`, `chroma_dir: Path`. Method: `ensure_dirs() -> None`.

- [ ] **Step 1: Install Python 3.11+**

```powershell
winget install --id Python.Python.3.12 -e --accept-source-agreements --accept-package-agreements
```

Close and reopen the shell so PATH refreshes, then verify:

```powershell
python --version
```

Expected: `Python 3.12.x` (any 3.11+ is acceptable). If `python` still does not resolve, the Microsoft Store alias may be shadowing it — disable it under Settings → Apps → Advanced app settings → App execution aliases.

- [ ] **Step 2: Create the virtual environment and install dependencies**

Create `requirements.txt`:

```
pymupdf>=1.24
sentence-transformers>=3.0
chromadb>=0.5.5
streamlit>=1.36
typer>=0.12
pydantic-settings>=2.3
pytest>=8.0
```

Then:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip freeze > requirements.lock.txt
```

Version floors are used rather than exact pins because the resolved set is not known ahead of time; `requirements.lock.txt` records what actually got installed so the environment is reproducible.

- [ ] **Step 3: Create the package skeleton and pytest config**

```powershell
mkdir ragforge, tests
New-Item -ItemType File ragforge\__init__.py, tests\__init__.py
```

Create `pytest.ini`:

```ini
[pytest]
testpaths = tests
markers =
    slow: requires the real embedding model or a real Chroma database
addopts = -q
```

- [ ] **Step 4: Write the failing test**

Create `tests/test_config.py`:

```python
from pathlib import Path

from ragforge.config import Settings


def test_defaults_match_spec():
    s = Settings()
    assert s.embedding_model_name == "BAAI/bge-small-en-v1.5"
    assert s.chunk_size == 400
    assert s.chunk_overlap == 60
    assert s.max_model_tokens == 510


def test_chunk_size_must_fit_the_model_window():
    import pytest
    with pytest.raises(ValueError):
        Settings(chunk_size=600)


def test_overlap_must_be_smaller_than_chunk_size():
    import pytest
    with pytest.raises(ValueError):
        Settings(chunk_size=100, chunk_overlap=100)


def test_derived_directories_hang_off_data_dir(tmp_path):
    s = Settings(data_dir=tmp_path)
    assert s.uploads_dir == tmp_path / "uploads"
    assert s.chroma_dir == tmp_path / "chroma"


def test_ensure_dirs_creates_them(tmp_path):
    s = Settings(data_dir=tmp_path / "d")
    s.ensure_dirs()
    assert s.uploads_dir.is_dir()
    assert s.chroma_dir.is_dir()


def test_env_override(monkeypatch):
    monkeypatch.setenv("RAGFORGE_CHUNK_SIZE", "250")
    assert Settings().chunk_size == 250
```

- [ ] **Step 5: Run the test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragforge.config'`

- [ ] **Step 6: Write the implementation**

Create `ragforge/config.py`:

```python
"""Typed settings for the ingestion pipeline."""
from __future__ import annotations

from pathlib import Path

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAGFORGE_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )

    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    # The model window is 512; the encoder adds [CLS] and [SEP], leaving 510 for text.
    max_model_tokens: int = 510
    chunk_size: int = 400
    chunk_overlap: int = 60
    embed_batch_size: int = 32
    collection_name: str = "documents"
    data_dir: Path = PROJECT_ROOT / "data"

    @field_validator("chunk_size", "chunk_overlap", "embed_batch_size")
    @classmethod
    def _must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be greater than zero")
        return v

    @model_validator(mode="after")
    def _check_relationships(self) -> "Settings":
        if self.chunk_size > self.max_model_tokens:
            raise ValueError(
                f"chunk_size {self.chunk_size} exceeds max_model_tokens "
                f"{self.max_model_tokens}; text would be silently truncated at embed time"
            )
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    def ensure_dirs(self) -> None:
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS — 6 passed.

- [ ] **Step 8: Commit**

```bash
git add requirements.txt requirements.lock.txt pytest.ini ragforge tests
git commit -m "feat: project scaffolding and typed settings"
```

---

### Task 2: PDF parsing and text normalization

**Files:**
- Create: `ragforge/parse.py`
- Create: `tests/conftest.py`
- Test: `tests/test_parse.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `PageText` frozen dataclass: `page_number: int` (1-indexed), `text: str`.
  - `PdfParseError(Exception)`, and subclasses `EncryptedPdfError`, `NoTextLayerError`, `CorruptPdfError`.
  - `normalize_text(raw: str) -> str`
  - `extract_pages(pdf_path: Path) -> list[PageText]` — normalized, empty pages dropped; raises the errors above.
  - `compute_doc_id(data: bytes) -> str` — full 64-char sha256 hex.

- [ ] **Step 1: Write the shared test fixtures**

Create `tests/conftest.py`:

```python
"""Fixtures shared across the test suite."""
from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF
import pytest


def _build_pdf(path: Path, pages: list[str], *, encrypt: bool = False) -> Path:
    """Write a small text PDF. One string per page."""
    doc = fitz.open()
    for body in pages:
        page = doc.new_page()
        page.insert_text((72, 72), body, fontsize=11)
    if encrypt:
        doc.save(
            path,
            encryption=fitz.PDF_ENCRYPT_AES_256,
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
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_parse.py`:

```python
import pytest

from ragforge.parse import (
    EncryptedPdfError,
    NoTextLayerError,
    CorruptPdfError,
    compute_doc_id,
    extract_pages,
    normalize_text,
)


def test_joins_words_broken_across_a_line():
    assert normalize_text("inter-\nnational") == "international"


def test_single_newlines_become_spaces():
    assert normalize_text("one\ntwo") == "one two"


def test_paragraph_breaks_survive():
    assert normalize_text("one\n\ntwo") == "one\n\ntwo"


def test_long_blank_runs_collapse_to_one_break():
    assert normalize_text("one\n\n\n\ntwo") == "one\n\ntwo"


def test_repeated_spaces_collapse():
    assert normalize_text("a     b") == "a b"


def test_extracts_one_entry_per_page_in_order(make_pdf):
    path = make_pdf(["alpha text", "beta text"])
    pages = extract_pages(path)
    assert [p.page_number for p in pages] == [1, 2]
    assert "alpha" in pages[0].text
    assert "beta" in pages[1].text


def test_blank_pages_are_dropped(make_pdf):
    path = make_pdf(["real content here", "   ", "more content"])
    pages = extract_pages(path)
    assert [p.page_number for p in pages] == [1, 3]


def test_encrypted_pdf_raises(make_pdf):
    path = make_pdf(["secret"], encrypt=True)
    with pytest.raises(EncryptedPdfError):
        extract_pages(path)


def test_pdf_with_no_text_raises(make_pdf):
    path = make_pdf(["", ""])
    with pytest.raises(NoTextLayerError):
        extract_pages(path)


def test_corrupt_file_raises(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"this is not a pdf")
    with pytest.raises(CorruptPdfError):
        extract_pages(path)


def test_doc_id_is_deterministic_and_content_sensitive():
    assert compute_doc_id(b"same") == compute_doc_id(b"same")
    assert compute_doc_id(b"same") != compute_doc_id(b"different")
    assert len(compute_doc_id(b"x")) == 64
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `pytest tests/test_parse.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragforge.parse'`

- [ ] **Step 4: Write the implementation**

Create `ragforge/parse.py`:

```python
"""Extract normalized, page-attributed text from digital PDFs."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


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
        doc = fitz.open(pdf_path)
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
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `pytest tests/test_parse.py -v`
Expected: PASS — 12 passed.

- [ ] **Step 6: Commit**

```bash
git add ragforge/parse.py tests/conftest.py tests/test_parse.py
git commit -m "feat: PDF text extraction with normalization and typed parse errors"
```

---

### Task 3: Token-aware chunking

The core algorithm. Recursive segmentation (paragraph → sentence → word → hard token cut) feeding a greedy packer that carries trailing segments forward as overlap.

**Files:**
- Create: `ragforge/chunk.py`
- Test: `tests/test_chunk.py`

**Interfaces:**
- Consumes: `ragforge.parse.PageText`.
- Produces:
  - `Tokenizer` Protocol: `count_tokens(text: str) -> int`, `split_by_tokens(text: str, max_tokens: int) -> list[str]`.
  - `Chunk` frozen dataclass: `id, doc_id, text, source_filename, page_start, page_end, chunk_index, token_count, chunk_size, overlap, ingested_at` (types as in the spec).
  - `chunk_id(doc_id: str, chunk_index: int) -> str`
  - `chunk_pages(pages, *, doc_id, source_filename, tokenizer, chunk_size, overlap, ingested_at) -> list[Chunk]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_chunk.py`:

```python
import pytest

from ragforge.chunk import Chunk, chunk_id, chunk_pages
from ragforge.parse import PageText

ISO = "2026-08-28T00:00:00+00:00"


def run(pages, tokenizer, chunk_size=10, overlap=3):
    return chunk_pages(
        pages,
        doc_id="doc",
        source_filename="f.pdf",
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        overlap=overlap,
        ingested_at=ISO,
    )


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_short_document_is_one_chunk(tokenizer):
    chunks = run([PageText(1, "alpha beta gamma")], tokenizer)
    assert len(chunks) == 1
    assert chunks[0].text == "alpha beta gamma"
    assert chunks[0].token_count == 3


def test_no_chunk_exceeds_chunk_size(tokenizer):
    chunks = run([PageText(1, words(95))], tokenizer, chunk_size=10, overlap=3)
    assert chunks, "expected chunks"
    assert all(c.token_count <= 10 for c in chunks)


def test_an_unbroken_run_longer_than_the_window_is_hard_split(tokenizer):
    # One "sentence" with no paragraph or sentence boundary anywhere.
    chunks = run([PageText(1, words(50))], tokenizer, chunk_size=10, overlap=0)
    assert all(c.token_count <= 10 for c in chunks)
    assert len(chunks) >= 5


def test_consecutive_chunks_overlap_but_do_not_exceed_the_budget(tokenizer):
    chunks = run([PageText(1, words(60))], tokenizer, chunk_size=10, overlap=3)
    assert len(chunks) >= 2
    for previous, current in zip(chunks, chunks[1:]):
        carried = set(previous.text.split()) & set(current.text.split())
        assert carried, "each chunk should re-include some of its predecessor"
        assert len(carried) <= 3


def test_zero_overlap_produces_disjoint_chunks(tokenizer):
    chunks = run([PageText(1, words(40))], tokenizer, chunk_size=10, overlap=0)
    for previous, current in zip(chunks, chunks[1:]):
        assert not (set(previous.text.split()) & set(current.text.split()))


def test_chunk_indexes_are_contiguous_from_zero(tokenizer):
    chunks = run([PageText(1, words(80))], tokenizer)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_page_provenance_for_a_chunk_spanning_two_pages(tokenizer):
    pages = [PageText(1, words(6, "a")), PageText(2, words(6, "b"))]
    chunks = run(pages, tokenizer, chunk_size=12, overlap=0)
    spanning = [c for c in chunks if c.page_start != c.page_end]
    assert spanning, "expected a chunk covering both pages"
    assert spanning[0].page_start == 1
    assert spanning[0].page_end == 2


def test_single_page_chunk_reports_that_page(tokenizer):
    chunks = run([PageText(7, "alpha beta")], tokenizer)
    assert chunks[0].page_start == 7
    assert chunks[0].page_end == 7


def test_paragraph_boundaries_are_preferred_over_mid_paragraph_splits(tokenizer):
    pages = [PageText(1, f"{words(8, 'p')}\n\n{words(8, 'q')}")]
    chunks = run(pages, tokenizer, chunk_size=10, overlap=0)
    # Neither chunk should mix the two paragraphs, since each fits alone.
    for c in chunks:
        assert not ({"p0"} & set(c.text.split()) and {"q0"} & set(c.text.split()))


def test_parameters_are_recorded_on_every_chunk(tokenizer):
    chunks = run([PageText(1, words(30))], tokenizer, chunk_size=10, overlap=3)
    assert all(c.chunk_size == 10 and c.overlap == 3 for c in chunks)
    assert all(c.ingested_at == ISO for c in chunks)


def test_chunk_ids_are_deterministic_and_unique(tokenizer):
    first = run([PageText(1, words(40))], tokenizer)
    second = run([PageText(1, words(40))], tokenizer)
    assert [c.id for c in first] == [c.id for c in second]
    assert len({c.id for c in first}) == len(first)


def test_chunk_id_depends_on_document_and_index():
    assert chunk_id("a", 0) != chunk_id("b", 0)
    assert chunk_id("a", 0) != chunk_id("a", 1)
    assert len(chunk_id("a", 0)) == 32


def test_empty_input_yields_no_chunks(tokenizer):
    assert run([], tokenizer) == []


def test_overlap_must_be_smaller_than_chunk_size(tokenizer):
    with pytest.raises(ValueError):
        run([PageText(1, "a b c")], tokenizer, chunk_size=5, overlap=5)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_chunk.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragforge.chunk'`

- [ ] **Step 3: Write the implementation**

Create `ragforge/chunk.py`:

```python
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


def _segment_page(page: PageText, tokenizer: Tokenizer, budget: int) -> list[_Segment]:
    """Break one page into segments that each fit the budget."""
    segments: list[_Segment] = []

    def emit(text: str) -> None:
        text = text.strip()
        if text:
            segments.append(
                _Segment(text=text, page_number=page.page_number,
                         token_count=tokenizer.count_tokens(text))
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
            for run in _split_words(sentence, tokenizer, budget):
                if tokenizer.count_tokens(run) <= budget:
                    emit(run)
                else:
                    # A single token-dense word: only a hard cut can bound it.
                    for piece in tokenizer.split_by_tokens(run, budget):
                        emit(piece)
    return segments


def _overlap_tail(segments: Sequence[_Segment], overlap: int) -> list[_Segment]:
    """Trailing segments of a chunk whose tokens total at most `overlap`."""
    if overlap <= 0:
        return []
    tail: list[_Segment] = []
    total = 0
    for segment in reversed(segments):
        if total + segment.token_count > overlap:
            break
        tail.insert(0, segment)
        total += segment.token_count
    return tail


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
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    segments: list[_Segment] = []
    for page in pages:
        segments.extend(_segment_page(page, tokenizer, chunk_size))
    if not segments:
        return []

    chunks: list[Chunk] = []
    current: list[_Segment] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        text = " ".join(s.text for s in current)
        index = len(chunks)
        chunks.append(
            Chunk(
                id=chunk_id(doc_id, index),
                doc_id=doc_id,
                text=text,
                source_filename=source_filename,
                page_start=min(s.page_number for s in current),
                page_end=max(s.page_number for s in current),
                chunk_index=index,
                token_count=tokenizer.count_tokens(text),
                chunk_size=chunk_size,
                overlap=overlap,
                ingested_at=ingested_at,
            )
        )
        carried = _overlap_tail(current, overlap)
        current = list(carried)
        current_tokens = sum(s.token_count for s in carried)

    for segment in segments:
        if current and current_tokens + segment.token_count > chunk_size:
            flush()
            # The carried tail plus this segment may still overflow; drop the
            # tail rather than exceed the budget.
            if current_tokens + segment.token_count > chunk_size:
                current, current_tokens = [], 0
        current.append(segment)
        current_tokens += segment.token_count

    # Final flush must not re-carry an overlap tail into a new empty chunk.
    if current:
        text = " ".join(s.text for s in current)
        index = len(chunks)
        chunks.append(
            Chunk(
                id=chunk_id(doc_id, index),
                doc_id=doc_id,
                text=text,
                source_filename=source_filename,
                page_start=min(s.page_number for s in current),
                page_end=max(s.page_number for s in current),
                chunk_index=index,
                token_count=tokenizer.count_tokens(text),
                chunk_size=chunk_size,
                overlap=overlap,
                ingested_at=ingested_at,
            )
        )

    return chunks
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_chunk.py -v`
Expected: PASS — 14 passed.

If `test_no_chunk_exceeds_chunk_size` fails, the bug is in the packer's overflow handling, not the segmenter — segments are guaranteed to fit by construction.

- [ ] **Step 5: Commit**

```bash
git add ragforge/chunk.py tests/test_chunk.py
git commit -m "feat: token-aware recursive chunker with page provenance and overlap"
```

---

### Task 4: Embedding and the real tokenizer

**Files:**
- Create: `ragforge/embed.py`
- Test: `tests/test_embed.py`

**Interfaces:**
- Consumes: `ragforge.chunk.Tokenizer` (satisfies the Protocol), `ragforge.config.settings`.
- Produces:
  - `HFTokenizer` — wraps a HuggingFace tokenizer, implements `Tokenizer`.
  - `Embedder` — `.tokenizer -> Tokenizer`, `.dimension -> int`, `.embed_documents(texts: Sequence[str], batch_size: int | None = None) -> list[list[float]]`, `.embed_query(text: str) -> list[float]`.
  - `FakeEmbedder` (in the test module, re-exported via `conftest`) is **not** part of this module.

Vectors are L2-normalized so cosine similarity is a dot product. Queries get the BGE retrieval instruction prefix; documents do not — that asymmetry is what the model was trained with, and skipping it measurably degrades retrieval.

- [ ] **Step 1: Write the failing test**

Create `tests/test_embed.py`:

```python
import pytest

from ragforge.embed import HFTokenizer, Embedder, QUERY_PREFIX


class _StubHF:
    """Minimal stand-in for a HuggingFace tokenizer."""

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"t{i}" for i in ids)


def test_counts_tokens_without_special_tokens():
    assert HFTokenizer(_StubHF()).count_tokens("a b c") == 3


def test_split_by_tokens_bounds_every_piece():
    pieces = HFTokenizer(_StubHF()).split_by_tokens("a b c d e", 2)
    assert len(pieces) == 3
    assert all(len(p.split()) <= 2 for p in pieces)


def test_query_prefix_is_the_bge_retrieval_instruction():
    assert QUERY_PREFIX.startswith("Represent this sentence")


@pytest.mark.slow
def test_real_model_round_trip():
    """Downloads the model. Run with: pytest -m slow"""
    embedder = Embedder()
    assert embedder.dimension == 384

    vectors = embedder.embed_documents(["the cat sat", "quarterly revenue report"])
    assert len(vectors) == 2
    assert all(len(v) == 384 for v in vectors)
    assert abs(sum(c * c for c in vectors[0]) ** 0.5 - 1.0) < 1e-3

    query = embedder.embed_query("where did the cat sit")
    cat_score = sum(a * b for a, b in zip(query, vectors[0]))
    revenue_score = sum(a * b for a, b in zip(query, vectors[1]))
    assert cat_score > revenue_score


@pytest.mark.slow
def test_tokenizer_agrees_with_the_model_window():
    embedder = Embedder()
    long_text = " ".join(["word"] * 5000)
    pieces = embedder.tokenizer.split_by_tokens(long_text, 510)
    assert all(embedder.tokenizer.count_tokens(p) <= 510 for p in pieces)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_embed.py -v -m "not slow"`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragforge.embed'`

- [ ] **Step 3: Write the implementation**

Create `ragforge/embed.py`:

```python
"""Local CPU embeddings and the tokenizer that bounds chunk sizes."""
from __future__ import annotations

from functools import cached_property
from typing import Sequence

from ragforge.config import Settings, settings as default_settings

# BGE models are trained with an asymmetric setup: queries carry a retrieval
# instruction, passages do not. Dropping this prefix measurably hurts recall.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class HFTokenizer:
    """Adapts a HuggingFace tokenizer to the chunker's Tokenizer Protocol."""

    def __init__(self, hf_tokenizer) -> None:
        self._tok = hf_tokenizer

    def count_tokens(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False))

    def split_by_tokens(self, text: str, max_tokens: int) -> list[str]:
        ids = self._tok.encode(text, add_special_tokens=False)
        return [
            self._tok.decode(ids[i : i + max_tokens], skip_special_tokens=True)
            for i in range(0, len(ids), max_tokens)
        ]


class Embedder:
    """Wraps a sentence-transformers model. Loads lazily on first use."""

    def __init__(self, config: Settings | None = None) -> None:
        self._settings = config or default_settings

    @cached_property
    def _model(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self._settings.embedding_model_name, device="cpu")

    @cached_property
    def tokenizer(self) -> HFTokenizer:
        return HFTokenizer(self._model.tokenizer)

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def embed_documents(
        self, texts: Sequence[str], batch_size: int | None = None
    ) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts),
            batch_size=batch_size or self._settings.embed_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(
            QUERY_PREFIX + text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_embed.py -v -m "not slow"`
Expected: PASS — 3 passed, 2 deselected.

- [ ] **Step 5: Verify against the real model once**

Run: `pytest tests/test_embed.py -v -m slow`
Expected: PASS — 2 passed. First run downloads ~130 MB; allow a few minutes.

- [ ] **Step 6: Commit**

```bash
git add ragforge/embed.py tests/test_embed.py
git commit -m "feat: local CPU embedder with BGE query prefix and HF tokenizer adapter"
```

---

### Task 5: Vector store

**Files:**
- Create: `ragforge/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: `ragforge.chunk.Chunk`, `ragforge.config.Settings`.
- Produces:
  - `Hit` frozen dataclass: `chunk_id, doc_id, text, source_filename, page_start, page_end, score`.
  - `DocumentSummary` frozen dataclass: `doc_id, source_filename, chunk_count, chunk_size, overlap, ingested_at`.
  - `VectorStore` Protocol: `upsert`, `query`, `delete_by_filename`, `get_document_params`, `count`, `list_documents`, `iter_chunks`.
  - `ChromaStore(persist_dir: Path, collection_name: str)` implementing it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_store.py`:

```python
import pytest

from ragforge.chunk import Chunk
from ragforge.store import ChromaStore

DIM = 4


def make_chunk(index=0, doc_id="doc1", filename="a.pdf", text="hello world"):
    return Chunk(
        id=f"{doc_id}-{index}",
        doc_id=doc_id,
        text=text,
        source_filename=filename,
        page_start=1,
        page_end=1,
        chunk_index=index,
        token_count=2,
        chunk_size=400,
        overlap=60,
        ingested_at="2026-08-28T00:00:00+00:00",
    )


def unit(*values):
    """A length-DIM vector, L2-normalized."""
    padded = list(values) + [0.0] * (DIM - len(values))
    norm = sum(v * v for v in padded) ** 0.5 or 1.0
    return [v / norm for v in padded]


@pytest.fixture
def store(tmp_path):
    return ChromaStore(persist_dir=tmp_path / "chroma", collection_name="test")


def test_starts_empty(store):
    assert store.count() == 0
    assert store.list_documents() == []


def test_upsert_then_count(store):
    store.upsert([make_chunk(0), make_chunk(1)], [unit(1, 0), unit(0, 1)])
    assert store.count() == 2


def test_query_returns_the_nearest_chunk_first(store):
    store.upsert(
        [make_chunk(0, text="cats"), make_chunk(1, text="ledgers")],
        [unit(1, 0), unit(0, 1)],
    )
    hits = store.query(unit(0.95, 0.05), k=2)
    assert hits[0].text == "cats"
    assert hits[0].score > hits[1].score


def test_scores_are_cosine_similarity_in_zero_to_one(store):
    store.upsert([make_chunk(0)], [unit(1, 0)])
    hit = store.query(unit(1, 0), k=1)[0]
    assert 0.99 <= hit.score <= 1.001


def test_hits_carry_provenance(store):
    store.upsert([make_chunk(0, filename="report.pdf")], [unit(1, 0)])
    hit = store.query(unit(1, 0), k=1)[0]
    assert hit.source_filename == "report.pdf"
    assert hit.page_start == 1
    assert hit.doc_id == "doc1"


def test_upserting_the_same_ids_replaces_rather_than_duplicates(store):
    store.upsert([make_chunk(0, text="first")], [unit(1, 0)])
    store.upsert([make_chunk(0, text="second")], [unit(1, 0)])
    assert store.count() == 1
    assert store.query(unit(1, 0), k=1)[0].text == "second"


def test_delete_by_filename_removes_only_that_document(store):
    store.upsert([make_chunk(0, doc_id="d1", filename="a.pdf")], [unit(1, 0)])
    store.upsert([make_chunk(0, doc_id="d2", filename="b.pdf")], [unit(0, 1)])
    removed = store.delete_by_filename("a.pdf")
    assert removed == 1
    assert store.count() == 1
    assert store.query(unit(0, 1), k=1)[0].source_filename == "b.pdf"


def test_get_document_params_reports_stored_settings(store):
    store.upsert([make_chunk(0, doc_id="d1")], [unit(1, 0)])
    assert store.get_document_params("d1") == {"chunk_size": 400, "overlap": 60}


def test_get_document_params_is_none_for_unknown_documents(store):
    assert store.get_document_params("nope") is None


def test_list_documents_groups_chunks(store):
    store.upsert(
        [make_chunk(0, doc_id="d1"), make_chunk(1, doc_id="d1")],
        [unit(1, 0), unit(0, 1)],
    )
    docs = store.list_documents()
    assert len(docs) == 1
    assert docs[0].chunk_count == 2
    assert docs[0].source_filename == "a.pdf"


def test_iter_chunks_round_trips_every_field(store):
    original = make_chunk(0)
    store.upsert([original], [unit(1, 0)])
    restored = list(store.iter_chunks())[0]
    assert restored == original


def test_iter_chunks_returns_documents_in_chunk_order(store):
    store.upsert(
        [make_chunk(1), make_chunk(0)],
        [unit(0, 1), unit(1, 0)],
    )
    assert [c.chunk_index for c in store.iter_chunks()] == [0, 1]


def test_persists_across_instances(tmp_path):
    path = tmp_path / "chroma"
    ChromaStore(path, "test").upsert([make_chunk(0)], [unit(1, 0)])
    assert ChromaStore(path, "test").count() == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragforge.store'`

- [ ] **Step 3: Write the implementation**

Create `ragforge/store.py`:

```python
"""Vector storage behind a narrow Protocol, implemented over ChromaDB."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol, Sequence

from ragforge.chunk import Chunk


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    doc_id: str
    text: str
    source_filename: str
    page_start: int
    page_end: int
    score: float  # cosine similarity, higher is better


@dataclass(frozen=True)
class DocumentSummary:
    doc_id: str
    source_filename: str
    chunk_count: int
    chunk_size: int
    overlap: int
    ingested_at: str


class VectorStore(Protocol):
    def upsert(
        self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]
    ) -> None: ...

    def query(self, vector: Sequence[float], k: int) -> list[Hit]: ...

    def delete_by_filename(self, filename: str) -> int: ...

    def get_document_params(self, doc_id: str) -> dict | None: ...

    def count(self) -> int: ...

    def list_documents(self) -> list[DocumentSummary]: ...

    def iter_chunks(self) -> Iterator[Chunk]: ...


_METADATA_FIELDS = (
    "doc_id",
    "source_filename",
    "page_start",
    "page_end",
    "chunk_index",
    "token_count",
    "chunk_size",
    "overlap",
    "ingested_at",
)


class ChromaStore:
    """Persistent ChromaDB collection. Vectors are supplied by us, never by Chroma."""

    def __init__(self, persist_dir: Path, collection_name: str = "documents") -> None:
        import chromadb
        from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE  # noqa: F401

        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]
    ) -> None:
        if not chunks:
            return
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")
        self._collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=[list(v) for v in vectors],
            documents=[c.text for c in chunks],
            metadatas=[{f: getattr(c, f) for f in _METADATA_FIELDS} for c in chunks],
        )

    def query(self, vector: Sequence[float], k: int) -> list[Hit]:
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[list(vector)],
            n_results=min(k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[Hit] = []
        for chunk_id, text, meta, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            hits.append(
                Hit(
                    chunk_id=chunk_id,
                    doc_id=meta["doc_id"],
                    text=text,
                    source_filename=meta["source_filename"],
                    page_start=int(meta["page_start"]),
                    page_end=int(meta["page_end"]),
                    # Chroma returns cosine distance; similarity is 1 - distance.
                    score=1.0 - float(distance),
                )
            )
        return hits

    def delete_by_filename(self, filename: str) -> int:
        existing = self._collection.get(
            where={"source_filename": filename}, include=[]
        )
        ids = existing["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def get_document_params(self, doc_id: str) -> dict | None:
        found = self._collection.get(
            where={"doc_id": doc_id}, limit=1, include=["metadatas"]
        )
        if not found["ids"]:
            return None
        meta = found["metadatas"][0]
        return {
            "chunk_size": int(meta["chunk_size"]),
            "overlap": int(meta["overlap"]),
        }

    def count(self) -> int:
        return int(self._collection.count())

    def list_documents(self) -> list[DocumentSummary]:
        summaries: dict[str, dict] = {}
        for chunk in self.iter_chunks():
            entry = summaries.setdefault(
                chunk.doc_id,
                {
                    "doc_id": chunk.doc_id,
                    "source_filename": chunk.source_filename,
                    "chunk_count": 0,
                    "chunk_size": chunk.chunk_size,
                    "overlap": chunk.overlap,
                    "ingested_at": chunk.ingested_at,
                },
            )
            entry["chunk_count"] += 1
        return [DocumentSummary(**entry) for entry in summaries.values()]

    def iter_chunks(self) -> Iterator[Chunk]:
        """Every chunk, ordered by document then chunk index."""
        records = self._collection.get(include=["documents", "metadatas"])
        rows = list(zip(records["ids"], records["documents"], records["metadatas"]))
        rows.sort(key=lambda r: (r[2]["doc_id"], int(r[2]["chunk_index"])))
        for chunk_id, text, meta in rows:
            yield Chunk(
                id=chunk_id,
                doc_id=meta["doc_id"],
                text=text,
                source_filename=meta["source_filename"],
                page_start=int(meta["page_start"]),
                page_end=int(meta["page_end"]),
                chunk_index=int(meta["chunk_index"]),
                token_count=int(meta["token_count"]),
                chunk_size=int(meta["chunk_size"]),
                overlap=int(meta["overlap"]),
                ingested_at=meta["ingested_at"],
            )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_store.py -v`
Expected: PASS — 13 passed.

If `include=[]` raises on your Chroma version, use `include=["metadatas"]` in `delete_by_filename` and keep only the ids.

- [ ] **Step 5: Commit**

```bash
git add ragforge/store.py tests/test_store.py
git commit -m "feat: ChromaDB vector store behind a VectorStore Protocol"
```

---

### Task 6: Pipeline orchestration

Wires everything together and owns the per-file error boundary, the re-ingestion rules, and atomicity.

**Files:**
- Create: `ragforge/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `parse.extract_pages`, `parse.compute_doc_id`, `parse` error classes, `chunk.chunk_pages`, `store.VectorStore`, `store.Hit`, `embed.Embedder`, `config.Settings`.
- Produces:
  - `FileResult` dataclass: `path: Path`, `status: str` (`"ingested" | "skipped" | "failed"`), `chunk_count: int`, `message: str`.
  - `IngestReport` dataclass: `results: list[FileResult]`, properties `ingested`, `skipped`, `failed` (lists), `total_chunks: int`.
  - `Pipeline(store, embedder, config)` with `ingest_file(path, *, chunk_size=None, overlap=None, force=False) -> FileResult`, `ingest_path(path, *, recursive=True, chunk_size=None, overlap=None, force=False) -> IngestReport`, `search(query, k=5) -> list[Hit]`, `stats() -> dict`.
  - `build_pipeline(config: Settings | None = None) -> Pipeline`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline.py`:

```python
import pytest

from ragforge.config import Settings
from ragforge.pipeline import Pipeline
from ragforge.store import ChromaStore
from tests.conftest import WhitespaceTokenizer


class FakeEmbedder:
    """Deterministic vectors, no model download. Dimension 8."""

    def __init__(self):
        self.tokenizer = WhitespaceTokenizer()
        self.calls = 0

    def _vector(self, text):
        vector = [0.0] * 8
        for word in text.split():
            vector[hash(word) % 8] += 1.0
        norm = sum(v * v for v in vector) ** 0.5 or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts, batch_size=None):
        self.calls += 1
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


@pytest.fixture
def pipeline(tmp_path):
    config = Settings(data_dir=tmp_path / "data", chunk_size=20, chunk_overlap=5)
    config.ensure_dirs()
    store = ChromaStore(config.chroma_dir, config.collection_name)
    return Pipeline(store=store, embedder=FakeEmbedder(), config=config)


def test_ingesting_a_pdf_reports_success_and_stores_chunks(pipeline, make_pdf):
    path = make_pdf(["alpha beta gamma delta", "epsilon zeta"])
    result = pipeline.ingest_file(path)
    assert result.status == "ingested"
    assert result.chunk_count > 0
    assert pipeline.store.count() == result.chunk_count


def test_the_source_pdf_is_copied_into_uploads(pipeline, make_pdf):
    path = make_pdf(["alpha beta"], name="report.pdf")
    pipeline.ingest_file(path)
    assert (pipeline.config.uploads_dir / "report.pdf").is_file()


def test_reingesting_identical_bytes_with_identical_params_is_skipped(pipeline, make_pdf):
    path = make_pdf(["alpha beta gamma"])
    first = pipeline.ingest_file(path)
    second = pipeline.ingest_file(path)
    assert second.status == "skipped"
    assert "already ingested" in second.message
    assert pipeline.store.count() == first.chunk_count


def test_reingesting_with_different_chunk_size_reprocesses(pipeline, make_pdf):
    path = make_pdf([" ".join(f"w{i}" for i in range(60))])
    pipeline.ingest_file(path, chunk_size=20, overlap=0)
    coarse = pipeline.store.count()
    result = pipeline.ingest_file(path, chunk_size=10, overlap=0)
    assert result.status == "ingested"
    assert pipeline.store.count() != coarse
    assert all(c.chunk_size == 10 for c in pipeline.store.iter_chunks())


def test_force_reingests_even_with_identical_params(pipeline, make_pdf):
    path = make_pdf(["alpha beta gamma"])
    pipeline.ingest_file(path)
    assert pipeline.ingest_file(path, force=True).status == "ingested"


def test_reingestion_leaves_no_orphan_chunks_from_the_old_version(pipeline, make_pdf):
    path = make_pdf([" ".join(f"w{i}" for i in range(60))], name="doc.pdf")
    pipeline.ingest_file(path, chunk_size=10, overlap=0)
    pipeline.ingest_file(path, chunk_size=20, overlap=0)
    assert len({c.chunk_size for c in pipeline.store.iter_chunks()}) == 1


def test_encrypted_pdf_fails_without_raising(pipeline, make_pdf):
    result = pipeline.ingest_file(make_pdf(["secret"], encrypt=True))
    assert result.status == "failed"
    assert "password" in result.message.lower()
    assert pipeline.store.count() == 0


def test_pdf_without_a_text_layer_fails_with_the_ocr_hint(pipeline, make_pdf):
    result = pipeline.ingest_file(make_pdf(["", ""]))
    assert result.status == "failed"
    assert "ocr" in result.message.lower()


def test_corrupt_file_fails_without_raising(pipeline, tmp_path):
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"not a pdf at all")
    assert pipeline.ingest_file(bad).status == "failed"


def test_nothing_is_written_when_a_file_fails(pipeline, make_pdf):
    pipeline.ingest_file(make_pdf(["good text here"]))
    before = pipeline.store.count()
    pipeline.ingest_file(make_pdf(["", ""]))
    assert pipeline.store.count() == before


def test_a_bad_file_does_not_abort_the_batch(pipeline, make_pdf, tmp_path):
    folder = tmp_path / "batch"
    folder.mkdir()
    make_pdf(["alpha beta"], name="batch/ok1.pdf")
    make_pdf(["gamma delta"], name="batch/ok2.pdf")
    (folder / "broken.pdf").write_bytes(b"garbage")

    report = pipeline.ingest_path(folder)
    assert len(report.ingested) == 2
    assert len(report.failed) == 1
    assert report.total_chunks > 0


def test_ingest_path_ignores_non_pdf_files(pipeline, tmp_path, make_pdf):
    folder = tmp_path / "mixed"
    folder.mkdir()
    make_pdf(["alpha beta"], name="mixed/doc.pdf")
    (folder / "notes.txt").write_text("ignore me")
    report = pipeline.ingest_path(folder)
    assert len(report.results) == 1


def test_ingest_path_accepts_a_single_file(pipeline, make_pdf):
    report = pipeline.ingest_path(make_pdf(["alpha beta"]))
    assert len(report.results) == 1


def test_ingest_path_recurses_when_asked(pipeline, tmp_path, make_pdf):
    nested = tmp_path / "top" / "inner"
    nested.mkdir(parents=True)
    make_pdf(["alpha beta"], name="top/inner/deep.pdf")
    assert len(pipeline.ingest_path(tmp_path / "top", recursive=True).results) == 1
    assert len(pipeline.ingest_path(tmp_path / "top", recursive=False).results) == 0


def test_search_finds_the_relevant_chunk(pipeline, make_pdf):
    pipeline.ingest_file(make_pdf(["quarterly revenue figures"], name="fin.pdf"))
    pipeline.ingest_file(make_pdf(["migratory bird patterns"], name="bio.pdf"))
    hits = pipeline.search("quarterly revenue figures", k=2)
    assert hits[0].source_filename == "fin.pdf"


def test_search_on_an_empty_store_returns_nothing(pipeline):
    assert pipeline.search("anything") == []


def test_stats_reports_documents_and_chunks(pipeline, make_pdf):
    pipeline.ingest_file(make_pdf(["alpha beta gamma"]))
    stats = pipeline.stats()
    assert stats["documents"] == 1
    assert stats["chunks"] == pipeline.store.count()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragforge.pipeline'`

- [ ] **Step 3: Write the implementation**

Create `ragforge/pipeline.py`:

```python
"""Orchestration. Contains no algorithms — only sequencing and error boundaries."""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ragforge.chunk import chunk_pages
from ragforge.config import Settings, settings as default_settings
from ragforge.parse import PdfParseError, compute_doc_id, extract_pages
from ragforge.store import Hit, VectorStore


@dataclass
class FileResult:
    path: Path
    status: str  # "ingested" | "skipped" | "failed"
    chunk_count: int = 0
    message: str = ""


@dataclass
class IngestReport:
    results: list[FileResult] = field(default_factory=list)

    @property
    def ingested(self) -> list[FileResult]:
        return [r for r in self.results if r.status == "ingested"]

    @property
    def skipped(self) -> list[FileResult]:
        return [r for r in self.results if r.status == "skipped"]

    @property
    def failed(self) -> list[FileResult]:
        return [r for r in self.results if r.status == "failed"]

    @property
    def total_chunks(self) -> int:
        return sum(r.chunk_count for r in self.results)


class Pipeline:
    def __init__(self, store: VectorStore, embedder, config: Settings | None = None):
        self.store = store
        self.embedder = embedder
        self.config = config or default_settings

    def ingest_file(
        self,
        path: Path,
        *,
        chunk_size: int | None = None,
        overlap: int | None = None,
        force: bool = False,
    ) -> FileResult:
        """Ingest one PDF. Never raises for bad input — returns a failed result."""
        path = Path(path)
        size = chunk_size if chunk_size is not None else self.config.chunk_size
        over = overlap if overlap is not None else self.config.chunk_overlap

        if size > self.config.max_model_tokens:
            return FileResult(path, "failed", 0,
                              f"chunk_size {size} exceeds the model window "
                              f"({self.config.max_model_tokens} tokens)")
        if over >= size:
            return FileResult(path, "failed", 0, "overlap must be smaller than chunk_size")

        try:
            data = path.read_bytes()
        except OSError as exc:
            return FileResult(path, "failed", 0, f"could not read file: {exc}")

        doc_id = compute_doc_id(data)

        if not force:
            stored = self.store.get_document_params(doc_id)
            if stored == {"chunk_size": size, "overlap": over}:
                return FileResult(path, "skipped", 0,
                                  "already ingested with these chunk parameters")

        try:
            pages = extract_pages(path)
        except PdfParseError as exc:
            return FileResult(path, "failed", 0, str(exc))
        except Exception as exc:  # defensive: one file must never kill a batch
            return FileResult(path, "failed", 0, f"unexpected parse failure: {exc}")

        chunks = chunk_pages(
            pages,
            doc_id=doc_id,
            source_filename=path.name,
            tokenizer=self.embedder.tokenizer,
            chunk_size=size,
            overlap=over,
            ingested_at=datetime.now(timezone.utc).isoformat(),
        )
        if not chunks:
            return FileResult(path, "failed", 0, "document produced no chunks")

        try:
            vectors = self.embedder.embed_documents(
                [c.text for c in chunks], batch_size=self.config.embed_batch_size
            )
        except Exception as exc:
            return FileResult(path, "failed", 0, f"embedding failed: {exc}")

        # Everything succeeded — only now touch the store, so an interrupted or
        # failed file can never leave a half-indexed document behind.
        self.store.delete_by_filename(path.name)
        self.store.upsert(chunks, vectors)

        self.config.ensure_dirs()
        destination = self.config.uploads_dir / path.name
        if path.resolve() != destination.resolve():
            shutil.copy2(path, destination)

        return FileResult(path, "ingested", len(chunks), f"{len(chunks)} chunks")

    def ingest_path(
        self,
        path: Path,
        *,
        recursive: bool = True,
        chunk_size: int | None = None,
        overlap: int | None = None,
        force: bool = False,
    ) -> IngestReport:
        path = Path(path)
        if path.is_dir():
            pattern = "**/*.pdf" if recursive else "*.pdf"
            files = sorted(p for p in path.glob(pattern) if p.is_file())
        else:
            files = [path]

        report = IngestReport()
        for file in files:
            report.results.append(
                self.ingest_file(
                    file, chunk_size=chunk_size, overlap=overlap, force=force
                )
            )
        return report

    def search(self, query: str, k: int = 5) -> list[Hit]:
        if not query.strip():
            return []
        return self.store.query(self.embedder.embed_query(query), k=k)

    def stats(self) -> dict:
        documents = self.store.list_documents()
        return {
            "documents": len(documents),
            "chunks": self.store.count(),
            "collection": self.config.collection_name,
            "chroma_dir": str(self.config.chroma_dir),
        }


def build_pipeline(config: Settings | None = None) -> Pipeline:
    """Wire the real ChromaStore and Embedder together."""
    from ragforge.embed import Embedder
    from ragforge.store import ChromaStore

    config = config or default_settings
    config.ensure_dirs()
    return Pipeline(
        store=ChromaStore(config.chroma_dir, config.collection_name),
        embedder=Embedder(config),
        config=config,
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS — 17 passed.

If `test_a_bad_file_does_not_abort_the_batch` fails on a missing directory, the `make_pdf` fixture from Task 2 is not creating parent directories for names like `"batch/ok1.pdf"` — add the `target.parent.mkdir(parents=True, exist_ok=True)` line shown there.

- [ ] **Step 5: Run the whole suite**

Run: `pytest -m "not slow" -v`
Expected: PASS — all unit tests green.

- [ ] **Step 6: Commit**

```bash
git add ragforge/pipeline.py tests/test_pipeline.py
git commit -m "feat: ingestion pipeline with per-file error boundaries and reingestion rules"
```

---

### Task 7: JSONL export and the CLI

**Files:**
- Create: `ragforge/export.py`
- Create: `cli.py`
- Test: `tests/test_export.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `store.VectorStore`, `pipeline.build_pipeline`, `pipeline.IngestReport`.
- Produces:
  - `export_jsonl(store: VectorStore, out_path: Path) -> int` — writes one object per line with keys `text`, `source`, `page_start`, `page_end`; returns the count written.
  - A Typer app `app` in `cli.py` with commands `ingest`, `search`, `stats`, `export`.

- [ ] **Step 1: Write the failing export test**

Create `tests/test_export.py`:

```python
import json

from ragforge.export import export_jsonl
from tests.test_store import make_chunk, unit

from ragforge.store import ChromaStore


def test_exports_one_json_object_per_chunk(tmp_path):
    store = ChromaStore(tmp_path / "chroma", "test")
    store.upsert(
        [make_chunk(0, text="first chunk"), make_chunk(1, text="second chunk")],
        [unit(1, 0), unit(0, 1)],
    )
    out = tmp_path / "corpus.jsonl"
    written = export_jsonl(store, out)

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert written == 2
    assert len(lines) == 2

    record = json.loads(lines[0])
    assert record == {
        "text": "first chunk",
        "source": "a.pdf",
        "page_start": 1,
        "page_end": 1,
    }


def test_exporting_an_empty_store_writes_an_empty_file(tmp_path):
    store = ChromaStore(tmp_path / "chroma", "test")
    out = tmp_path / "corpus.jsonl"
    assert export_jsonl(store, out) == 0
    assert out.read_text(encoding="utf-8") == ""


def test_export_handles_non_ascii(tmp_path):
    store = ChromaStore(tmp_path / "chroma", "test")
    store.upsert([make_chunk(0, text="café — naïve")], [unit(1, 0)])
    out = tmp_path / "corpus.jsonl"
    export_jsonl(store, out)
    assert json.loads(out.read_text(encoding="utf-8"))["text"] == "café — naïve"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_export.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragforge.export'`

- [ ] **Step 3: Write the export implementation**

Create `ragforge/export.py`:

```python
"""Dump the corpus as JSONL — the raw-text shape continued pretraining consumes."""
from __future__ import annotations

import json
from pathlib import Path

from ragforge.store import VectorStore


def export_jsonl(store: VectorStore, out_path: Path) -> int:
    """Write one JSON object per chunk. Returns the number of records written."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in store.iter_chunks():
            handle.write(
                json.dumps(
                    {
                        "text": chunk.text,
                        "source": chunk.source_filename,
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
    return written
```

- [ ] **Step 4: Verify the export test passes**

Run: `pytest tests/test_export.py -v`
Expected: PASS — 3 passed.

- [ ] **Step 5: Write the failing CLI test**

Create `tests/test_cli.py`:

```python
from typer.testing import CliRunner

import cli as cli_module
from ragforge.config import Settings
from ragforge.pipeline import Pipeline
from ragforge.store import ChromaStore
from tests.test_pipeline import FakeEmbedder

runner = CliRunner()


def _install_test_pipeline(monkeypatch, tmp_path):
    config = Settings(data_dir=tmp_path / "data", chunk_size=20, chunk_overlap=5)
    config.ensure_dirs()
    pipeline = Pipeline(
        store=ChromaStore(config.chroma_dir, config.collection_name),
        embedder=FakeEmbedder(),
        config=config,
    )
    monkeypatch.setattr(cli_module, "build_pipeline", lambda config=None: pipeline)
    return pipeline


def test_ingest_reports_per_file_status(monkeypatch, tmp_path, make_pdf):
    _install_test_pipeline(monkeypatch, tmp_path)
    pdf = make_pdf(["alpha beta gamma"], name="doc.pdf")
    result = runner.invoke(cli_module.app, ["ingest", str(pdf)])
    assert result.exit_code == 0
    assert "ingested" in result.stdout
    assert "doc.pdf" in result.stdout


def test_ingest_exits_nonzero_when_every_file_fails(monkeypatch, tmp_path):
    _install_test_pipeline(monkeypatch, tmp_path)
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"garbage")
    result = runner.invoke(cli_module.app, ["ingest", str(bad)])
    assert result.exit_code == 1
    assert "failed" in result.stdout


def test_search_prints_ranked_hits(monkeypatch, tmp_path, make_pdf):
    pipeline = _install_test_pipeline(monkeypatch, tmp_path)
    pipeline.ingest_file(make_pdf(["quarterly revenue figures"], name="fin.pdf"))
    result = runner.invoke(cli_module.app, ["search", "quarterly revenue figures"])
    assert result.exit_code == 0
    assert "fin.pdf" in result.stdout


def test_stats_prints_counts(monkeypatch, tmp_path, make_pdf):
    pipeline = _install_test_pipeline(monkeypatch, tmp_path)
    pipeline.ingest_file(make_pdf(["alpha beta"]))
    result = runner.invoke(cli_module.app, ["stats"])
    assert result.exit_code == 0
    assert "documents" in result.stdout.lower()


def test_export_writes_the_file(monkeypatch, tmp_path, make_pdf):
    pipeline = _install_test_pipeline(monkeypatch, tmp_path)
    pipeline.ingest_file(make_pdf(["alpha beta"]))
    out = tmp_path / "corpus.jsonl"
    result = runner.invoke(cli_module.app, ["export", str(out)])
    assert result.exit_code == 0
    assert out.is_file()
    assert out.read_text(encoding="utf-8").strip()
```

- [ ] **Step 6: Run it to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cli'`

- [ ] **Step 7: Write the CLI**

Create `cli.py`:

```python
"""Command line entrypoint. Thin wrapper over ragforge.pipeline."""
from __future__ import annotations

from pathlib import Path

import typer

from ragforge.export import export_jsonl
from ragforge.pipeline import build_pipeline

app = typer.Typer(help="Local PDF ingestion and retrieval.", no_args_is_help=True)


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="A PDF file or a directory of PDFs."),
    chunk_size: int = typer.Option(None, help="Tokens per chunk."),
    overlap: int = typer.Option(None, help="Tokens of overlap between chunks."),
    recursive: bool = typer.Option(True, help="Descend into subdirectories."),
    force: bool = typer.Option(False, help="Re-ingest even if already stored."),
) -> None:
    """Ingest a PDF or a folder of PDFs."""
    pipeline = build_pipeline()
    report = pipeline.ingest_path(
        path, recursive=recursive, chunk_size=chunk_size, overlap=overlap, force=force
    )

    if not report.results:
        typer.echo("No PDF files found.")
        raise typer.Exit(code=1)

    for result in report.results:
        typer.echo(f"[{result.status:>8}] {result.path.name} — {result.message}")

    typer.echo(
        f"\n{len(report.ingested)} ingested, {len(report.skipped)} skipped, "
        f"{len(report.failed)} failed, {report.total_chunks} chunks written."
    )
    if report.failed and not report.ingested:
        raise typer.Exit(code=1)


@app.command()
def search(
    query: str = typer.Argument(..., help="What to look for."),
    k: int = typer.Option(5, help="Number of results."),
) -> None:
    """Search the corpus and print ranked chunks."""
    hits = build_pipeline().search(query, k=k)
    if not hits:
        typer.echo("No results.")
        return
    for rank, hit in enumerate(hits, start=1):
        pages = (
            f"p{hit.page_start}"
            if hit.page_start == hit.page_end
            else f"p{hit.page_start}-{hit.page_end}"
        )
        typer.echo(f"\n{rank}. {hit.score:.3f}  {hit.source_filename}  {pages}")
        typer.echo(f"   {hit.text[:300]}")


@app.command()
def stats() -> None:
    """Show what is currently in the store."""
    for key, value in build_pipeline().stats().items():
        typer.echo(f"{key}: {value}")


@app.command()
def export(
    out_path: Path = typer.Argument(..., help="Destination .jsonl file."),
) -> None:
    """Export every chunk as JSONL for later training use."""
    written = export_jsonl(build_pipeline().store, out_path)
    typer.echo(f"Wrote {written} records to {out_path}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 8: Verify the CLI test passes**

Run: `pytest tests/test_cli.py -v`
Expected: PASS — 5 passed.

- [ ] **Step 9: Commit**

```bash
git add ragforge/export.py cli.py tests/test_export.py tests/test_cli.py
git commit -m "feat: JSONL export and Typer CLI"
```

---

### Task 8: Streamlit interface

**Files:**
- Create: `app.py`
- Create: `ragforge/ui_helpers.py`
- Test: `tests/test_ui_helpers.py`

**Interfaces:**
- Consumes: `pipeline.build_pipeline`, `pipeline.IngestReport`, `store.Hit`, `store.DocumentSummary`.
- Produces: `ui_helpers.format_page_range(page_start, page_end) -> str`, `ui_helpers.report_rows(report) -> list[dict]`, `ui_helpers.hit_rows(hits) -> list[dict]`.

Streamlit callbacks are hard to test directly, so the pure formatting logic lives in `ui_helpers.py` and is unit-tested; `app.py` stays declarative layout only.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ui_helpers.py`:

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_ui_helpers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ragforge.ui_helpers'`

- [ ] **Step 3: Write the helpers**

Create `ragforge/ui_helpers.py`:

```python
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
```

- [ ] **Step 4: Verify the test passes**

Run: `pytest tests/test_ui_helpers.py -v`
Expected: PASS — 4 passed.

- [ ] **Step 5: Write the Streamlit app**

Create `app.py`:

```python
"""Streamlit interface: ingest, inspect, search."""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from ragforge.config import settings
from ragforge.pipeline import build_pipeline
from ragforge.ui_helpers import format_page_range, hit_rows, report_rows

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
        from ragforge.pipeline import IngestReport

        st.dataframe(report_rows(IngestReport(results=results)),
                     use_container_width=True, hide_index=True)
        st.rerun()

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
                    path, recursive=recursive, chunk_size=chunk_size,
                    overlap=overlap, force=force,
                )
            st.dataframe(report_rows(report), use_container_width=True,
                         hide_index=True)
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
```

- [ ] **Step 6: Launch and verify by hand**

Run: `streamlit run app.py`

Verify each of these in the browser:
1. Sidebar shows Documents 0 / Chunks 0 on a fresh store.
2. Uploading a PDF produces a results table with status `ingested` and a chunk count.
3. The sidebar counts increase after ingestion.
4. The Inspect tab lists the document and expands chunks showing page range and token count.
5. Every chunk's token count is at or below the sidebar's chunk size.
6. A search for a phrase from the PDF returns it as the top hit with the right page number.
7. Re-uploading the same PDF at the same settings reports `skipped`.
8. Changing the chunk size slider and re-uploading reports `ingested` and changes the chunk count.

- [ ] **Step 7: Commit**

```bash
git add app.py ragforge/ui_helpers.py tests/test_ui_helpers.py
git commit -m "feat: Streamlit ingest, inspect and search interface"
```

---

### Task 9: End-to-end integration test and README

Proves the real components work together with the real model and a real database.

**Files:**
- Create: `tests/test_integration.py`
- Create: `README.md`

**Interfaces:**
- Consumes: everything.
- Produces: nothing new.

- [ ] **Step 1: Write the integration test**

Create `tests/test_integration.py`:

```python
"""Full-stack tests using the real embedding model and a real Chroma database.

Run with: pytest -m slow
"""
import pytest

from ragforge.config import Settings
from ragforge.embed import Embedder
from ragforge.export import export_jsonl
from ragforge.pipeline import Pipeline
from ragforge.store import ChromaStore

pytestmark = pytest.mark.slow

PAGE_ONE = (
    "The quarterly financial report covers revenue, operating margin, and "
    "headcount changes across all business units for the period."
)
PAGE_TWO = (
    "Migratory bird populations along the northern flyway were surveyed using "
    "banding data collected over twelve consecutive breeding seasons."
)


@pytest.fixture
def real_pipeline(tmp_path):
    config = Settings(data_dir=tmp_path / "data")
    config.ensure_dirs()
    return Pipeline(
        store=ChromaStore(config.chroma_dir, config.collection_name),
        embedder=Embedder(config),
        config=config,
    )


def test_ingest_then_search_returns_the_right_page(real_pipeline, make_pdf):
    pdf = make_pdf([PAGE_ONE, PAGE_TWO], name="mixed.pdf")
    result = real_pipeline.ingest_file(pdf)
    assert result.status == "ingested"

    hits = real_pipeline.search("how many birds were counted in the survey", k=2)
    assert hits, "expected at least one hit"
    assert hits[0].page_start == 2
    assert "bird" in hits[0].text.lower()

    hits = real_pipeline.search("what was the operating margin", k=2)
    assert hits[0].page_start == 1


def test_reingesting_the_same_file_does_not_grow_the_store(real_pipeline, make_pdf):
    pdf = make_pdf([PAGE_ONE], name="stable.pdf")
    real_pipeline.ingest_file(pdf)
    count_after_first = real_pipeline.store.count()
    real_pipeline.ingest_file(pdf)
    assert real_pipeline.store.count() == count_after_first


def test_forced_reingestion_keeps_the_count_stable(real_pipeline, make_pdf):
    pdf = make_pdf([PAGE_ONE], name="stable.pdf")
    real_pipeline.ingest_file(pdf)
    count_after_first = real_pipeline.store.count()
    real_pipeline.ingest_file(pdf, force=True)
    assert real_pipeline.store.count() == count_after_first


def test_every_stored_chunk_fits_the_model_window(real_pipeline, make_pdf):
    body = " ".join([PAGE_ONE, PAGE_TWO] * 20)
    real_pipeline.ingest_file(make_pdf([body], name="long.pdf"))
    tokenizer = real_pipeline.embedder.tokenizer
    for chunk in real_pipeline.store.iter_chunks():
        assert tokenizer.count_tokens(chunk.text) <= real_pipeline.config.max_model_tokens


def test_export_covers_every_stored_chunk(real_pipeline, make_pdf, tmp_path):
    real_pipeline.ingest_file(make_pdf([PAGE_ONE, PAGE_TWO], name="mixed.pdf"))
    out = tmp_path / "corpus.jsonl"
    assert export_jsonl(real_pipeline.store, out) == real_pipeline.store.count()
```

- [ ] **Step 2: Run the integration test**

Run: `pytest tests/test_integration.py -v -m slow`
Expected: PASS — 5 passed. Slower than the unit suite; the model loads once per test via the fixture.

- [ ] **Step 3: Run the complete suite**

Run: `pytest -v`
Expected: PASS — every test, unit and slow, green.

- [ ] **Step 4: Write the README**

Create `README.md`:

````markdown
# RAGForge

Local PDF ingestion for RAG. Upload or point at PDFs, watch them become
token-bounded chunks in a vector database, inspect what chunking produced, and
search the result. Everything runs on this machine — no API keys, no cloud.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The first run downloads the embedding model (`BAAI/bge-small-en-v1.5`, ~130 MB).

## Use

Web interface:

```powershell
streamlit run app.py
```

Command line:

```powershell
python cli.py ingest C:\path\to\pdfs --chunk-size 400 --overlap 60
python cli.py search "what was the operating margin"
python cli.py stats
python cli.py export corpus.jsonl
```

## Configuration

Any setting can be overridden with a `RAGFORGE_`-prefixed environment variable
or a `.env` file:

| Variable | Default | Meaning |
|---|---|---|
| `RAGFORGE_CHUNK_SIZE` | 400 | Tokens per chunk |
| `RAGFORGE_CHUNK_OVERLAP` | 60 | Tokens carried between chunks |
| `RAGFORGE_EMBEDDING_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `RAGFORGE_DATA_DIR` | `./data` | Uploads and database location |

## Tuning chunk size

Chunk size is corpus-dependent. Ingest a representative document, open the
Inspect tab, and read the chunks: if related sentences are split apart, raise
the size; if chunks contain several unrelated topics, lower it. Re-ingesting
the same file at different settings replaces its chunks rather than duplicating
them, so iterating is safe.

No chunk may exceed 510 tokens — the model's window minus its special tokens.
Beyond that, text is silently truncated at embed time and the stored text stops
matching its own vector.

## Tests

```powershell
pytest -m "not slow"   # fast: no model download, no database
pytest                 # everything, including the real model
```

## Not included

OCR for scanned PDFs, LLM answer generation, reranking, table extraction, and
authentication. Scanned PDFs are reported clearly rather than failing silently.

## Exporting for model training

`python cli.py export corpus.jsonl` writes one JSON object per chunk
(`text`, `source`, `page_start`, `page_end`) — the raw-corpus shape continued
pretraining consumes. Training itself is a separate project.
````

- [ ] **Step 5: Commit**

```bash
git add tests/test_integration.py README.md
git commit -m "test: end-to-end integration coverage; docs: README"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| Environment constraints, Python install | 1 |
| Technology decisions, config | 1 |
| PDF parsing, normalization | 2 |
| `compute_doc_id` determinism | 2 |
| Chunking strategy, token invariant, page provenance | 3 |
| Embeddings, tokenizer, CPU-only | 4 |
| Data model persistence, cosine similarity | 5 |
| `chunk_size`/`overlap` stored per chunk | 5 |
| Ingest flow incl. re-ingest and `--force` rules | 6 |
| Error handling, atomicity, batch resilience | 6 |
| Search flow | 6 |
| Export hook | 7 |
| CLI (`ingest`/`search`/`stats`/`export`) | 7 |
| Streamlit ingest/inspect/search | 8 |
| Testing strategy (unit + integration + error paths) | 2–9 |
| Setup documentation | 9 |

No spec requirement is unassigned.

**Type consistency:** `Tokenizer` (`count_tokens`, `split_by_tokens`) is defined in Task 3 and implemented by `HFTokenizer` (Task 4) and the test `WhitespaceTokenizer` (Task 2). `Chunk` fields are identical in Tasks 3, 5, and 7. `VectorStore` methods used by `Pipeline` (Task 6) and `export_jsonl` (Task 7) all exist on `ChromaStore` (Task 5). `Hit` fields used in Tasks 6–8 match Task 5's definition. `IngestReport`/`FileResult` used in Tasks 7–8 match Task 6.

**Known cross-task coupling:** `tests/test_cli.py` and `tests/test_export.py` import helpers from `tests/test_pipeline.py` and `tests/test_store.py`. If the implementer prefers, promote `FakeEmbedder`, `make_chunk`, and `unit` into `tests/conftest.py` — behavior is unchanged either way.
