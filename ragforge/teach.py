"""The 3H teaching layer: retrieve, generate, then verify what came back.

Retrieval always returns the closest available passages, even when the corpus
holds nothing relevant. That makes verification part of the contract rather
than a nicety: citations are checked against what was actually retrieved, and
a vector claiming coverage without any citation is downgraded.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from ragforge.config import Settings, settings as default_settings
from ragforge.llm import LLMError, LLMResponse, OllamaClient
from ragforge.store import Hit

VECTORS = ("head", "heart", "hands")

VECTOR_BADGES = {
    "head": "HEAD",
    "heart": "HEART",
    "hands": "HANDS",
}

VECTOR_LABELS = {
    "head": "HEAD — cognitive",
    "heart": "HEART — affective",
    "hands": "HANDS — psychomotor",
}

_CARD_SCHEMA = {
    "type": "object",
    "properties": {
        "vector": {"type": "string", "enum": list(VECTORS)},
        "headline": {"type": "string"},
        "bullets": {"type": "array", "items": {"type": "string"}},
        "citations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["vector", "headline", "bullets", "citations"],
}

TEACH_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "overview": {"type": "string"},
        "cards": {"type": "array", "items": _CARD_SCHEMA},
        "picture_this": {"type": "string"},
        "gap_report": {"type": "array", "items": {"type": "string"}},
        "unverified_claims": {"type": "array", "items": {"type": "string"}},
        "retrieval_question": {"type": "string"},
    },
    "required": [
        "title", "overview", "cards", "picture_this",
        "gap_report", "unverified_claims", "retrieval_question",
    ],
}


def citation_label(hit: Hit) -> str:
    """The label a passage is cited by: "ram.pdf p5" or "ram.pdf p5-7"."""
    if hit.page_start == hit.page_end:
        return f"{hit.source_filename} p{hit.page_start}"
    return f"{hit.source_filename} p{hit.page_start}-{hit.page_end}"


def build_context(hits: Sequence[Hit]) -> str:
    """The CONTEXT block. Each passage is prefixed with its citation label."""
    return "\n\n".join(f"[{citation_label(h)}]\n{h.text}" for h in hits)


def _normalize_citation(raw: str) -> str:
    """Strip brackets and collapse whitespace so comparison is about content."""
    return re.sub(r"\s+", " ", raw.strip().strip("[]").strip())


@dataclass
class TeachingCard:
    """One teaching point: a claim, its supporting facts, and its sources."""

    vector: str
    headline: str
    bullets: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    dropped_citations: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return VECTOR_LABELS[self.vector]

    @property
    def grounded(self) -> bool:
        """Whether any claim here traces to a retrieved passage."""
        return bool(self.citations)


@dataclass
class TeachAnswer:
    question: str
    title: str
    overview: str
    cards: list[TeachingCard]
    picture_this: str
    gap_report: list[str]
    unverified_claims: list[str]
    retrieval_question: str
    hits: list[Hit]
    warnings: list[str] = field(default_factory=list)
    llm: LLMResponse | None = None

    def cards_for(self, vector: str) -> list[TeachingCard]:
        return [c for c in self.cards if c.vector == vector]

    @property
    def covered_vectors(self) -> list[str]:
        """A vector is covered when it has at least one grounded card."""
        return [
            v for v in VECTORS if any(c.grounded for c in self.cards_for(v))
        ]

    @property
    def uncovered_vectors(self) -> list[str]:
        return [v for v in VECTORS if v not in self.covered_vectors]

    @property
    def sources(self) -> list[str]:
        """Every cited passage, in the order the vectors are taught."""
        seen: list[str] = []
        for card in self.cards:
            for citation in card.citations:
                if citation not in seen:
                    seen.append(citation)
        return seen


def load_system_prompt(config: Settings | None = None) -> str:
    """The 3H spec plus the deployment addendum, concatenated."""
    config = config or default_settings
    directory = Path(config.prompts_dir)
    parts = []
    for name in ("3h_agent.md", "3h_deployment.md"):
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"missing prompt file: {path}")
        parts.append(path.read_text(encoding="utf-8"))
    return "\n\n---\n\n".join(parts)


def _parse_card(raw: dict, allowed: set[str]) -> TeachingCard | None:
    """Build a card, discarding citations that were never retrieved.

    Returns None for a card with no vector or no headline — a malformed entry
    is dropped rather than rendered as an empty box.
    """
    vector = str(raw.get("vector") or "").strip().lower()
    if vector not in VECTORS:
        return None
    headline = (raw.get("headline") or "").strip()
    if not headline:
        return None

    kept, dropped = [], []
    for citation in raw.get("citations") or []:
        normalized = _normalize_citation(str(citation))
        if not normalized:
            continue  # models sometimes emit [""] rather than []
        (kept if normalized in allowed else dropped).append(normalized)

    return TeachingCard(
        vector=vector,
        headline=headline,
        bullets=[str(b).strip() for b in (raw.get("bullets") or []) if str(b).strip()],
        citations=kept,
        dropped_citations=dropped,
    )


class Teacher:
    """Answers a question from the corpus in the 3H format."""

    def __init__(
        self,
        pipeline,
        client: OllamaClient | None = None,
        config: Settings | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.config = config or default_settings
        self.client = client or OllamaClient(
            base_url=self.config.ollama_base_url,
            model=self.config.ollama_model,
            timeout_seconds=self.config.ollama_timeout_seconds,
            temperature=self.config.ollama_temperature,
        )

    def answer(self, question: str, k: int | None = None) -> TeachAnswer:
        """Retrieve, teach, and verify. Raises LLMError if generation fails."""
        if not question.strip():
            raise ValueError("question must not be empty")

        k = k or self.config.teach_context_chunks
        hits = self.pipeline.search(question, k=k)
        if not hits:
            raise LLMError(
                "nothing in the corpus matches this question — ingest relevant "
                "documents first"
            )

        allowed = {citation_label(h) for h in hits}
        user_message = (
            f"CONTEXT:\n{build_context(hits)}\n\n"
            f"LEARNER QUESTION: {question.strip()}"
        )

        parsed, meta = self.client.chat_json(
            system=load_system_prompt(self.config),
            user=user_message,
            schema=TEACH_SCHEMA,
        )

        cards: list[TeachingCard] = []
        for raw_card in parsed.get("cards") or []:
            card = _parse_card(raw_card if isinstance(raw_card, dict) else {}, allowed)
            if card is not None:
                cards.append(card)

        # Teach in the framework's own order, whatever order the model emitted.
        cards.sort(key=lambda c: VECTORS.index(c.vector))

        warnings: list[str] = []
        for card in cards:
            if card.dropped_citations:
                warnings.append(
                    f"{card.vector}: discarded citation(s) not in the retrieved "
                    f"passages — {', '.join(card.dropped_citations)}"
                )
            # R0.1: a claim that traces to nothing is ungrounded, whatever the
            # model asserts. Surfaced rather than silently dropped — the reader
            # decides, but is told.
            if not card.grounded:
                warnings.append(
                    f"{card.vector}: \"{card.headline[:60]}\" has no citation "
                    f"traceable to the retrieved passages"
                )

        return TeachAnswer(
            question=question.strip(),
            title=(parsed.get("title") or question).strip(),
            overview=(parsed.get("overview") or "").strip(),
            cards=cards,
            picture_this=(parsed.get("picture_this") or "").strip(),
            gap_report=[str(g).strip() for g in (parsed.get("gap_report") or []) if str(g).strip()],
            unverified_claims=[
                str(c).strip() for c in (parsed.get("unverified_claims") or []) if str(c).strip()
            ],
            retrieval_question=(parsed.get("retrieval_question") or "").strip(),
            hits=list(hits),
            warnings=warnings,
            llm=meta,
        )
