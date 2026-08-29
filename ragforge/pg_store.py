"""PostgreSQL + pgvector implementation of the VectorStore protocol.

Interchangeable with ChromaStore: same seven methods, same semantics, same
cosine-similarity scoring. Nothing outside this file knows which one is in use.
"""
from __future__ import annotations

from typing import Iterator, Sequence

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from ragforge.chunk import Chunk
from ragforge.store import DocumentSummary, Hit

_COLUMNS = (
    "id, doc_id, text, source_filename, page_start, page_end, "
    "chunk_index, token_count, chunk_size, overlap, ingested_at"
)


def _row_to_chunk(row: dict) -> Chunk:
    return Chunk(
        id=row["id"],
        doc_id=row["doc_id"],
        text=row["text"],
        source_filename=row["source_filename"],
        page_start=row["page_start"],
        page_end=row["page_end"],
        chunk_index=row["chunk_index"],
        token_count=row["token_count"],
        chunk_size=row["chunk_size"],
        overlap=row["overlap"],
        ingested_at=row["ingested_at"],
    )


class PgVectorStore:
    """Persistent vector store backed by PostgreSQL with the pgvector extension.

    Opens one connection and keeps it. The application is single-user and the
    operations are short, so a pool would add moving parts for no benefit.
    """

    def __init__(self, database_url: str, dimension: int = 384) -> None:
        self._dimension = dimension
        self._conn = psycopg.connect(database_url, autocommit=True)
        # The extension must exist before the vector type can be registered on
        # the connection, and both must precede any DDL using vector(N).
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self._conn)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create the table and indexes if the database is fresh.

        The container runs sql/001_schema.sql on first init, but a database
        pointed at by RAGFORGE_DATABASE_URL may not have been through that, so
        this makes the store self-sufficient.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id              TEXT PRIMARY KEY,
                    doc_id          TEXT        NOT NULL,
                    text            TEXT        NOT NULL,
                    source_filename TEXT        NOT NULL,
                    page_start      INTEGER     NOT NULL,
                    page_end        INTEGER     NOT NULL,
                    chunk_index     INTEGER     NOT NULL,
                    token_count     INTEGER     NOT NULL,
                    chunk_size      INTEGER     NOT NULL,
                    overlap         INTEGER     NOT NULL,
                    ingested_at     TEXT        NOT NULL,
                    embedding       vector({self._dimension}) NOT NULL
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw "
                "ON chunks USING hnsw (embedding vector_cosine_ops)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS chunks_source_filename "
                "ON chunks (source_filename)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS chunks_doc_id ON chunks (doc_id)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS chunks_doc_order "
                "ON chunks (doc_id, chunk_index)"
            )

        self._check_dimension()

    def _check_dimension(self) -> None:
        """Fail loudly if the existing table's vector width is not what we expect.

        CREATE TABLE IF NOT EXISTS silently keeps the old column definition, so
        a table built for a different embedding model would otherwise only fail
        deep inside an insert with an opaque error. This is the situation after
        switching models: the stored vectors are meaningless for the new model
        and the corpus has to be re-ingested.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT format_type(atttypid, atttypmod)
                FROM pg_attribute
                WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'
                """
            )
            row = cur.fetchone()
        if not row:
            return
        declared = row[0]  # e.g. "vector(384)"
        if not declared.startswith("vector(") or not declared.endswith(")"):
            return
        actual = int(declared[len("vector(") : -1])
        if actual != self._dimension:
            raise ValueError(
                f"the chunks table stores vector({actual}) but this store was "
                f"opened for vector({self._dimension}). The existing corpus was "
                f"embedded with a different model and cannot be searched with "
                f"this one. Re-ingest the documents into an empty table, or set "
                f"RAGFORGE_EMBEDDING_DIMENSION={actual}."
            )

    def close(self) -> None:
        if not self._conn.closed:
            self._conn.close()

    def upsert(
        self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]
    ) -> None:
        if not chunks:
            return
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")

        rows = [
            (
                c.id, c.doc_id, c.text, c.source_filename, c.page_start,
                c.page_end, c.chunk_index, c.token_count, c.chunk_size,
                c.overlap, c.ingested_at, list(v),
            )
            for c, v in zip(chunks, vectors)
        ]
        with self._conn.cursor() as cur:
            # Re-ingesting a document must replace its chunks, never duplicate
            # them. Chunk ids are deterministic, so the primary key collides and
            # this updates in place.
            cur.executemany(
                f"""
                INSERT INTO chunks ({_COLUMNS}, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    doc_id          = EXCLUDED.doc_id,
                    text            = EXCLUDED.text,
                    source_filename = EXCLUDED.source_filename,
                    page_start      = EXCLUDED.page_start,
                    page_end        = EXCLUDED.page_end,
                    chunk_index     = EXCLUDED.chunk_index,
                    token_count     = EXCLUDED.token_count,
                    chunk_size      = EXCLUDED.chunk_size,
                    overlap         = EXCLUDED.overlap,
                    ingested_at     = EXCLUDED.ingested_at,
                    embedding       = EXCLUDED.embedding
                """,
                rows,
            )

    def query(self, vector: Sequence[float], k: int) -> list[Hit]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            # <=> is pgvector's cosine distance. Similarity is 1 - distance,
            # matching the convention ChromaStore returns.
            cur.execute(
                f"""
                SELECT {_COLUMNS}, embedding <=> %s::vector AS distance
                FROM chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (list(vector), list(vector), k),
            )
            return [
                Hit(
                    chunk_id=row["id"],
                    doc_id=row["doc_id"],
                    text=row["text"],
                    source_filename=row["source_filename"],
                    page_start=row["page_start"],
                    page_end=row["page_end"],
                    score=1.0 - float(row["distance"]),
                )
                for row in cur.fetchall()
            ]

    def delete_by_filename(self, filename: str) -> int:
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE source_filename = %s", (filename,))
            return cur.rowcount

    def get_document_params(self, doc_id: str) -> dict | None:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT chunk_size, overlap FROM chunks WHERE doc_id = %s LIMIT 1",
                (doc_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return {"chunk_size": row["chunk_size"], "overlap": row["overlap"]}

    def count(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks")
            return int(cur.fetchone()[0])

    def list_documents(self) -> list[DocumentSummary]:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT doc_id,
                       MIN(source_filename) AS source_filename,
                       COUNT(*)             AS chunk_count,
                       MIN(chunk_size)      AS chunk_size,
                       MIN(overlap)         AS overlap,
                       MIN(ingested_at)     AS ingested_at
                FROM chunks
                GROUP BY doc_id
                ORDER BY MIN(source_filename)
                """
            )
            return [
                DocumentSummary(
                    doc_id=row["doc_id"],
                    source_filename=row["source_filename"],
                    chunk_count=int(row["chunk_count"]),
                    chunk_size=int(row["chunk_size"]),
                    overlap=int(row["overlap"]),
                    ingested_at=row["ingested_at"],
                )
                for row in cur.fetchall()
            ]

    def iter_chunks(self) -> Iterator[Chunk]:
        """Every chunk, ordered by document then chunk index."""
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                f"SELECT {_COLUMNS} FROM chunks ORDER BY doc_id, chunk_index"
            )
            for row in cur.fetchall():
                yield _row_to_chunk(row)
