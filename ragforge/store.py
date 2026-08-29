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
        total = self.count()
        if total == 0:
            return []
        result = self._collection.query(
            query_embeddings=[list(vector)],
            n_results=min(k, total),
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
