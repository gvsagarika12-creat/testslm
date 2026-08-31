-- RAGForge vector store schema.
-- Runs automatically the first time the Postgres container initialises.

CREATE EXTENSION IF NOT EXISTS vector;

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
    -- Dimension must match the embedding model. bge-small-en-v1.5 is 384.
    -- Changing models means changing this column and re-embedding everything.
    embedding       vector(384) NOT NULL
);

-- Approximate nearest-neighbour index for cosine distance. Without it, search
-- degrades to a full table scan as the corpus grows.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- delete_by_filename and get_document_params filter on these.
CREATE INDEX IF NOT EXISTS chunks_source_filename ON chunks (source_filename);
CREATE INDEX IF NOT EXISTS chunks_doc_id          ON chunks (doc_id);

-- iter_chunks returns rows in this order.
CREATE INDEX IF NOT EXISTS chunks_doc_order ON chunks (doc_id, chunk_index);
