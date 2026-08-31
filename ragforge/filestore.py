"""Where uploaded source files are kept.

The pipeline stores every ingested file so the original is recoverable after
chunking. Today that is a folder on this machine; tomorrow it may be an FTP
server, S3 bucket, or network share. Putting it behind a Protocol means the
pipeline never learns which, exactly as VectorStore does for the database.

To add a backend, implement the four methods below and register it in
`build_file_store`. Nothing else in the codebase changes.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Protocol


class FileStoreError(Exception):
    """Storing a file failed. Ingestion treats this as a per-file failure."""


class FileStore(Protocol):
    def save(self, filename: str, data: bytes) -> str:
        """Store `data` under `filename`. Returns a URI identifying it."""
        ...

    def exists(self, filename: str) -> bool: ...

    def read(self, filename: str) -> bytes: ...

    def list_files(self) -> list[str]: ...

    @property
    def location(self) -> str:
        """Human-readable description of where files go, for the UI."""
        ...


class LocalFileStore:
    """Files kept in a directory on this machine."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, filename: str) -> Path:
        # Never let a crafted name escape the root.
        name = Path(filename).name
        if not name or name in {".", ".."}:
            raise FileStoreError(f"invalid filename: {filename!r}")
        return self._root / name

    def save(self, filename: str, data: bytes) -> str:
        target = self._path(filename)
        try:
            # Write to a temp file in the same directory, then replace
            # atomically. A crash mid-write cannot leave a truncated file, and
            # re-saving a file over itself is safe because the bytes are
            # already in memory.
            fd, tmp = tempfile.mkstemp(dir=self._root, suffix=".part")
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
                os.replace(tmp, target)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
        except OSError as exc:
            raise FileStoreError(f"could not store {filename!r}: {exc}") from exc
        return target.resolve().as_uri()

    def exists(self, filename: str) -> bool:
        return self._path(filename).is_file()

    def read(self, filename: str) -> bytes:
        try:
            return self._path(filename).read_bytes()
        except OSError as exc:
            raise FileStoreError(f"could not read {filename!r}: {exc}") from exc

    def list_files(self) -> list[str]:
        return sorted(p.name for p in self._root.iterdir() if p.is_file())

    @property
    def location(self) -> str:
        return str(self._root)


def build_file_store(config) -> FileStore:
    """Construct the file store named by config.file_store_backend.

    An FTP backend belongs here: a class implementing save/exists/read/
    list_files over ftplib, returning "ftp://host/path/name" from save().
    """
    if config.file_store_backend == "local":
        return LocalFileStore(config.uploads_dir)
    raise ValueError(f"unknown file store backend: {config.file_store_backend!r}")
