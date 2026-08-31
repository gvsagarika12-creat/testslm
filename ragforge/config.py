"""Typed settings for the ingestion pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

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

    # Which vector store backs the corpus. "postgres" is the default; "chroma"
    # remains available as a no-server fallback for anyone without Docker.
    # Where uploaded source files are kept. "ftp" and friends slot in here
    # without the pipeline changing; see ragforge/filestore.py.
    file_store_backend: Literal["local"] = "local"

    store_backend: Literal["postgres", "chroma"] = "postgres"
    database_url: str = "postgresql://ragforge:ragforge@127.0.0.1:5432/ragforge"
    # Must equal the embedding model's output size and the vector(N) column.
    embedding_dimension: int = 384

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
