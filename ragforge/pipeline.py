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
            return FileResult(
                path,
                "failed",
                0,
                f"chunk_size {size} exceeds the model window "
                f"({self.config.max_model_tokens} tokens)",
            )
        if over >= size:
            return FileResult(
                path, "failed", 0, "overlap must be smaller than chunk_size"
            )

        try:
            data = path.read_bytes()
        except OSError as exc:
            return FileResult(path, "failed", 0, f"could not read file: {exc}")

        doc_id = compute_doc_id(data)

        if not force:
            stored = self.store.get_document_params(doc_id)
            if stored == {"chunk_size": size, "overlap": over}:
                return FileResult(
                    path, "skipped", 0, "already ingested with these chunk parameters"
                )

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
        info = {
            "documents": len(documents),
            "chunks": self.store.count(),
            "backend": self.config.store_backend,
        }
        if self.config.store_backend == "postgres":
            # Never print the password.
            url = self.config.database_url
            info["location"] = url.rsplit("@", 1)[-1] if "@" in url else url
        else:
            info["location"] = str(self.config.chroma_dir)
            info["collection"] = self.config.collection_name
        return info


def build_store(config: Settings) -> VectorStore:
    """Construct the vector store named by config.store_backend."""
    if config.store_backend == "postgres":
        from ragforge.pg_store import PgVectorStore

        return PgVectorStore(config.database_url, config.embedding_dimension)

    from ragforge.store import ChromaStore

    return ChromaStore(config.chroma_dir, config.collection_name)


def build_pipeline(config: Settings | None = None) -> Pipeline:
    """Wire the configured store and the Embedder together."""
    from ragforge.embed import Embedder

    config = config or default_settings
    config.ensure_dirs()
    return Pipeline(
        store=build_store(config),
        embedder=Embedder(config),
        config=config,
    )
