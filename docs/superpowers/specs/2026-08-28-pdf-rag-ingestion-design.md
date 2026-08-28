# PDF RAG Ingestion Pipeline — Design

**Date:** 2026-08-28
**Status:** Approved
**Scope:** Ingestion + retrieval only. SLM training is a separate future spec.

## Problem

We need a local system that takes PDFs, splits them into retrievable chunks, embeds
them, and stores them in a vector database. The immediate purpose is building and
tuning a RAG system. A later, separate project will use the same corpus to train a
small language model, so the chunk store must be exportable as training data.

## Goals

- Upload a PDF through a browser interface and see it become searchable chunks.
- Ingest a folder of PDFs from the command line in one invocation.
- Inspect what chunking actually produced, and tune chunk parameters against that feedback.
- Search the corpus and see ranked results with source provenance (filename, page).
- Export the corpus as JSONL for later training use.

## Non-Goals

Deliberately excluded to keep this shippable. Each is a documented extension point,
not an oversight.

- Authentication, user accounts, multi-tenancy.
- OCR for scanned PDFs. Corpus is born-digital text.
- LLM answer generation. Retrieval quality must be correct before generation is added.
- Reranking, hybrid/BM25 search, query rewriting.
- Table and figure structure extraction.
- Background job queue or async ingestion.
- Cloud APIs of any kind. Everything runs on the local machine.

## Environment Constraints

- Windows 11. Node 24 and Docker are present; **Python is not installed** and must be
  added (3.11+) as the first setup step.
- GPU is a Quadro P600 with 2 GB VRAM. Too small for meaningful model training and not
  needed here: embedding runs on CPU. This constraint is the reason SLM training is
  out of scope.

## Technology Decisions

| Concern | Choice | Reason |
|---|---|---|
| Language | Python 3.11+ | Standard RAG ecosystem; required by the later SLM phase. |
| PDF parsing | PyMuPDF | Fast, accurate text extraction with per-page boundaries. |
| Embeddings | `BAAI/bge-small-en-v1.5` via sentence-transformers | 384-dim, ~130 MB, CPU-viable, 512-token window, stronger than MiniLM. |
| Vector DB | ChromaDB (embedded, persistent) | No container to run; adequate to ~100k chunks; hidden behind a Protocol so it can be swapped. |
| UI | Streamlit | Drag-drop upload, chunk inspector, and search in minimal code. |
| CLI | Typer | Folder ingestion and scripted use. |
| Config | pydantic-settings | Typed settings from env/`.env` with defaults. |
| Tests | pytest | — |

## Architecture

A core library with two thin entrypoints. Both entrypoints call the same pipeline
functions, so UI and CLI behavior cannot diverge.

```
ragforge/
  config.py     Settings: chunk size, overlap, model name, data paths
  parse.py      Path -> list[PageText]          (PyMuPDF)
  chunk.py      list[PageText] -> list[Chunk]   (pure, token-aware)
  embed.py      list[str] -> list[vector]       (sentence-transformers)
  store.py      VectorStore Protocol + ChromaStore implementation
  pipeline.py   ingest(path) -> IngestReport; search(query, k) -> list[Hit]
  export.py     corpus -> JSONL
app.py          Streamlit interface
cli.py          ingest | search | stats | export
tests/
data/
  uploads/      copies of ingested source PDFs
  chroma/       Chroma persistence directory
```

### Module boundaries

- `parse`, `chunk`, and `embed` are pure with respect to the database. They take data
  and return data, touching no store. This makes them unit-testable in isolation and
  keeps the expensive, hard-to-test I/O confined to `store` and `pipeline`.
- `store.py` defines a `VectorStore` Protocol (`upsert`, `query`, `delete_document`,
  `count`, `iter_all`) with `ChromaStore` as the only implementation. Replacing Chroma
  with Qdrant later means adding one file, not editing callers.
- `pipeline.py` contains orchestration and no algorithms. It is the single API that
  `app.py` and `cli.py` consume.

## Data Model

```python
@dataclass(frozen=True)
class Chunk:
    id: str                 # sha256(doc_id + chunk_index)[:32]
    doc_id: str             # sha256 of raw file bytes
    text: str
    source_filename: str
    page_start: int         # 1-indexed, inclusive
    page_end: int           # 1-indexed, inclusive
    chunk_index: int        # ordinal within the document
    token_count: int
    chunk_size: int         # parameters this chunk was produced with
    overlap: int
    ingested_at: str        # ISO 8601 UTC
```

`chunk_size` and `overlap` are stored per chunk so ingestion can tell whether an
already-ingested document was chunked with the parameters now being requested, and
so the inspector can label which settings produced the chunks on screen.

**Both identifiers are deterministic.** `doc_id` is the hash of the file's bytes, so
the same PDF ingested twice is recognized as the same document. `id` is derived from
`doc_id` and the chunk ordinal, so re-ingestion upserts over the previous chunks
rather than appending duplicates. Duplicate chunks silently degrade retrieval by
crowding result sets with identical text, and the corruption is invisible until
someone inspects the store — preventing it up front costs almost nothing.

If a document's content changes, its bytes change, so its `doc_id` changes and it is
treated as a new document. `pipeline.ingest` deletes any prior chunks sharing the same
`source_filename` before upserting, so an edited-and-re-ingested file does not leave
orphaned chunks from its old version behind.

## Chunking Strategy

Recursive splitting that prefers the largest natural boundary that fits: paragraph
break, then sentence end, then word boundary, then hard character cut as a last resort.

Sizes are measured in **tokens using the embedding model's own tokenizer**, not in
characters. `bge-small-en-v1.5` truncates input beyond 512 tokens silently. A chunk
measured in characters can exceed that limit, in which case the stored text and the
text the vector actually represents are different — retrieval degrades with no error
and no obvious cause. The chunker therefore enforces `token_count <= model_max_tokens`
as a hard invariant, asserted in tests.

**Defaults:** 400 tokens per chunk, 60 tokens of overlap. Both are exposed as controls
in the Streamlit UI and as CLI flags, because tuning them against real documents is a
primary purpose of this tool.

Page provenance is preserved through splitting: a chunk spanning a page break records
`page_start` and `page_end` covering the range it came from.

**Normalization before chunking:** collapse repeated whitespace, join words broken by
hyphen-plus-newline at line ends, and drop pages that are empty after normalization.

## Data Flow

**Ingest**

1. Read file bytes, compute `doc_id`.
2. Consult the store for an existing entry with this `doc_id`. Skip and report
   "already ingested" **only if the stored chunk parameters match the requested
   ones**. If the requested chunk size or overlap differs, proceed — the document is
   re-chunked and its old chunks replaced. Tuning chunk parameters against a fixed
   document is a primary workflow, so identical bytes alone must not block re-ingestion.
   A `--force` flag re-ingests unconditionally.
3. Extract text per page with PyMuPDF.
4. Normalize; if the document yields no text at all, abort this file with
   "no text layer — likely scanned, OCR not enabled".
5. Chunk with page provenance.
6. Embed in batches of 32.
7. Delete any existing chunks for this `source_filename`, then upsert the new ones.
8. Copy the source PDF to `data/uploads/`.
9. Return an `IngestReport` for the file.

**Search**

Query text -> embed -> `store.query(vector, k)` -> hits carrying similarity score,
chunk text, filename, and page range, ordered by score.

## Error Handling

Batch ingestion is resilient by design: one unreadable file must not abort a run of
two hundred.

- Every file is processed inside its own error boundary. Failures are captured into
  the per-file report and the batch continues.
- Encrypted or password-protected PDFs: skipped with an explicit reason.
- Corrupt or non-PDF files: skipped with the parser's error message.
- Zero extractable text: skipped with the scanned-document message above. Empty chunks
  are never written.
- Ingestion of a single file is atomic. Chunks are upserted only after the whole file
  has been parsed, chunked, and embedded successfully, so an interruption cannot leave
  a partially indexed document.
- Both the UI and the CLI print the full per-file report: succeeded, skipped, failed,
  each with counts and reasons.

## Interfaces

### Streamlit (`app.py`)

Three sections on one page.

1. **Ingest** — file uploader (multi-file) plus a text box for a folder path. Chunk
   size and overlap sliders. Ingest button. Results table showing per-file status,
   chunk count, and any error.
2. **Inspect** — pick an ingested document, page through its chunks, see text,
   token count, and page range. This is how chunk parameters get evaluated.
3. **Search** — query box, top-k selector, ranked results with score, source filename,
   page range, and the matching text.

### CLI (`cli.py`)

```
ragforge ingest <path>          file or directory; --chunk-size, --overlap, --recursive
ragforge search "<query>"       --k
ragforge stats                  document count, chunk count, store size
ragforge export <out.jsonl>     dump corpus as JSONL
```

## Export Hook

`export.py` writes one JSON object per line: `{"text", "source", "page_start",
"page_end"}`. This is the raw-corpus shape that continued pretraining consumes. No
instruction-pair synthesis, no formatting for a specific trainer — that belongs in the
SLM spec, which will be written when a corpus and suitable hardware exist.

## Testing Strategy

Unit tests (no model download, no database — the embedder is stubbed with a fake
returning fixed-dimension vectors):

- Chunker respects the token limit for adversarial inputs, including a single
  unbroken run of text longer than the window.
- Overlap between consecutive chunks is the configured number of tokens.
- Chunk ordering and `chunk_index` continuity.
- Page provenance is correct for chunks spanning a page break.
- `doc_id` and chunk `id` are deterministic across runs and differ for differing bytes.
- Normalization joins hyphenated line breaks and collapses whitespace.
- Parsing a small PDF fixture generated at test time yields expected per-page text.

Integration test (temp directory, real Chroma, small real model, marked `slow`):

- Ingest a fixture PDF, search for a phrase known to be in it, assert the top hit is
  the correct chunk with the correct page number.
- Ingest the same file twice; assert the chunk count does not change.

Error-path tests: encrypted PDF, corrupt file, and a text-free PDF each produce the
correct report entry without raising.

## Setup

1. Install Python 3.11+ and confirm `python` resolves on PATH.
2. `python -m venv .venv` and activate.
3. `pip install -r requirements.txt`.
4. First run downloads the embedding model (~130 MB) to the HuggingFace cache.
5. `streamlit run app.py`, or use `ragforge` from the CLI.

## Risks

- **Chunk parameters are corpus-dependent.** The defaults are a starting point, not an
  answer. The inspector exists specifically so they get tuned against real documents.
- **Chroma's ceiling.** Fine to roughly 100k chunks. Past that, or if metadata
  filtering becomes central, migrate to Qdrant behind the existing `VectorStore`
  Protocol.
- **Embedding is CPU-bound.** Ingestion of a large backlog will be slow. Acceptable:
  ingestion is infrequent and batched.
- **No OCR.** A scanned PDF is reported clearly rather than failing silently, but it
  will not be ingested.
