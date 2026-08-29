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

Start the database (PostgreSQL 16 with pgvector, in Docker):

```powershell
docker compose up -d
```

The first app run downloads the embedding model (`BAAI/bge-small-en-v1.5`, ~130 MB).

The database must be running before the app starts. `docker compose down` stops
it and keeps your data; `docker compose down -v` deletes the stored vectors.

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

## How it works

```
PDF ──▶ parse ──▶ chunk ──▶ embed ──▶ PostgreSQL + pgvector
        (PyMuPDF)  (token-   (bge-small,
                    aware)     CPU)
```

`ragforge/` holds the core library — `parse`, `chunk`, and `embed` are pure and
never touch the database, so they test in isolation. `pipeline.py` is the only
API the interfaces use, so the UI and CLI cannot drift apart.

Storage sits behind a `VectorStore` Protocol with two interchangeable
implementations: `pg_store.PgVectorStore` (PostgreSQL + pgvector, the default)
and `store.ChromaStore` (embedded ChromaDB, no server required). The same
behavioural test suite runs against both, so they cannot diverge. Select one
with `RAGFORGE_STORE_BACKEND`.

## Configuration

Any setting can be overridden with a `RAGFORGE_`-prefixed environment variable
or a `.env` file:

| Variable | Default | Meaning |
|---|---|---|
| `RAGFORGE_CHUNK_SIZE` | 400 | Tokens per chunk |
| `RAGFORGE_CHUNK_OVERLAP` | 60 | Tokens carried between chunks |
| `RAGFORGE_EMBEDDING_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `RAGFORGE_DATA_DIR` | `./data` | Uploaded PDFs (and the Chroma store, if used) |
| `RAGFORGE_STORE_BACKEND` | `postgres` | `postgres` or `chroma` |
| `RAGFORGE_DATABASE_URL` | `postgresql://ragforge:ragforge@127.0.0.1:5432/ragforge` | Postgres connection |
| `RAGFORGE_EMBEDDING_DIMENSION` | 384 | Must match the model and the `vector(N)` column |

To run without Docker, set `RAGFORGE_STORE_BACKEND=chroma` — everything works
identically, storing to `data/chroma/` with no server.

### Migrating an existing ChromaDB corpus

```powershell
python cli.py migrate
```

Chunks move with their stored vectors, so nothing is re-embedded.

## Tuning chunk size

Chunk size is corpus-dependent. Ingest a representative document, open the
Inspect tab, and read the chunks: if related sentences are split apart, raise
the size; if chunks contain several unrelated topics, lower it. Re-ingesting the
same file at different settings replaces its chunks rather than duplicating
them, so iterating is safe — and re-ingesting at the *same* settings is skipped,
so nothing is recomputed for free.

No chunk may exceed 510 tokens — the model's 512-token window minus its two
special tokens. Beyond that, text is silently truncated at embed time and the
stored text stops matching its own vector.

## Tests

```powershell
pytest -m "not slow"   # fast: no model download, no database
pytest                 # everything, including the real model
```

## Not included

OCR for scanned PDFs, LLM answer generation, reranking, table extraction, and
authentication. A scanned PDF is reported as *"no text layer — likely a scanned
document, OCR not enabled"* rather than failing silently or storing empty chunks.

## Exporting for model training

`python cli.py export corpus.jsonl` writes one JSON object per chunk
(`text`, `source`, `page_start`, `page_end`) — the raw-corpus shape continued
pretraining consumes. Training itself is a separate project; note that it needs
considerably more GPU memory than this machine's 2 GB.

## Design documents

- Spec: [docs/superpowers/specs/2026-08-28-pdf-rag-ingestion-design.md](docs/superpowers/specs/2026-08-28-pdf-rag-ingestion-design.md)
- Plan: [docs/superpowers/plans/2026-08-28-pdf-rag-ingestion.md](docs/superpowers/plans/2026-08-28-pdf-rag-ingestion.md)
