"""Copy an existing ChromaDB corpus into PostgreSQL.

Chunks are moved with their stored vectors, so nothing is re-embedded and the
migration takes seconds rather than minutes. Chunk ids are deterministic and
the Postgres upsert is keyed on them, so running this twice is harmless.
"""
from __future__ import annotations

from ragforge.config import Settings, settings as default_settings
from ragforge.store import ChromaStore


def migrate_chroma_to_postgres(
    config: Settings | None = None, batch_size: int = 500
) -> int:
    """Move every chunk from the Chroma store into Postgres. Returns the count."""
    from ragforge.pg_store import PgVectorStore

    config = config or default_settings
    source = ChromaStore(config.chroma_dir, config.collection_name)
    target = PgVectorStore(config.database_url, config.embedding_dimension)

    # Chroma stores the vector alongside the chunk; read both so the embedding
    # model is never loaded.
    records = source._collection.get(include=["embeddings", "metadatas", "documents"])
    by_id = dict(zip(records["ids"], records["embeddings"]))

    moved = 0
    batch_chunks, batch_vectors = [], []
    for chunk in source.iter_chunks():
        vector = by_id.get(chunk.id)
        if vector is None:
            continue
        batch_chunks.append(chunk)
        batch_vectors.append([float(v) for v in vector])
        if len(batch_chunks) >= batch_size:
            target.upsert(batch_chunks, batch_vectors)
            moved += len(batch_chunks)
            batch_chunks, batch_vectors = [], []

    if batch_chunks:
        target.upsert(batch_chunks, batch_vectors)
        moved += len(batch_chunks)

    target.close()
    return moved
