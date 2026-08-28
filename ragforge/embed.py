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

    def tail_tokens(self, text: str, max_tokens: int) -> str:
        """The last `max_tokens` tokens of `text`, decoded back to a string."""
        if max_tokens <= 0:
            return ""
        ids = self._tok.encode(text, add_special_tokens=False)
        return self._tok.decode(ids[-max_tokens:], skip_special_tokens=True)


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
        # Renamed in sentence-transformers 6; keep working on older releases too.
        getter = getattr(self._model, "get_embedding_dimension", None) or (
            self._model.get_sentence_embedding_dimension
        )
        return int(getter())

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
