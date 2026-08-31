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

VECTOR_LABELS = {
    "head": "HEAD — cognitive",
    "heart": "HEART — affective",
    "hands": "HANDS — psychomotor",
}

_SECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "covered": {"type": "boolean"},
        "content": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["covered", "content", "citations"],
}

TEACH_SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {"type": "string"},
        "head": _SECTION_SCHEMA,
        "heart": _SECTION_SCHEMA,
        "hands": _SECTION_SCHEMA,
        "gap_report": {"type": "array", "items": {"type": "string"}},
        "unverified_claims": {"type": "array", "items": {"type": "string"}},
        "retrieval_question": {"type": "string"},
    },
    "required": [
        "topic", "head", "heart", "hands",
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
class VectorSection:
    vector: str
    covered: bool
    content: str
    citations: list[str] = field(default_factory=list)
    dropped_citations: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return VECTOR_LABELS[self.vector]


@dataclass
class TeachAnswer:
    question: str
    topic: str
    sections: dict[str, VectorSection]
    gap_report: list[str]
    unverified_claims: list[str]
    retrieval_question: str
    hits: list[Hit]
    warnings: list[str] = field(default_factory=list)
    llm: LLMResponse | None = None

    @property
    def covered_vectors(self) -> list[str]:
        return [v for v in VECTORS if self.sections[v].covered]

    @property
    def uncovered_vectors(self) -> list[str]:
        return [v for v in VECTORS if not self.sections[v].covered]


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


def _parse_section(vector: str, raw: dict, allowed: set[str]) -> VectorSection:
    """Build a section, discarding citations that were never retrieved."""
    kept, dropped = [], []
    for citation in raw.get("citations") or []:
        normalized = _normalize_citation(str(citation))
        if not normalized:
            continue  # models sometimes emit [""] for an empty section
        (kept if normalized in allowed else dropped).append(normalized)

    return VectorSection(
        vector=vector,
        covered=bool(raw.get("covered")),
        content=(raw.get("content") or "").strip(),
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

        sections = {
            v: _parse_section(v, parsed.get(v) or {}, allowed) for v in VECTORS
        }
        warnings: list[str] = []

        for section in sections.values():
            if section.dropped_citations:
                warnings.append(
                    f"{section.vector}: discarded citation(s) not in the retrieved "
                    f"passages — {', '.join(section.dropped_citations)}"
                )
            # A vector claiming coverage with no verifiable citation is not
            # grounded, whatever it says. R0.1 makes that a contract violation,
            # so demote it rather than presenting it as sourced.
            if section.covered and section.content and not section.citations:
                section.covered = False
                warnings.append(
                    f"{section.vector}: marked uncovered — content had no citation "
                    f"traceable to the retrieved passages"
                )

        return TeachAnswer(
            question=question.strip(),
            topic=(parsed.get("topic") or question).strip(),
            sections=sections,
            gap_report=[str(g).strip() for g in (parsed.get("gap_report") or []) if str(g).strip()],
            unverified_claims=[
                str(c).strip() for c in (parsed.get("unverified_claims") or []) if str(c).strip()
            ],
            retrieval_question=(parsed.get("retrieval_question") or "").strip(),
            hits=list(hits),
            warnings=warnings,
            llm=meta,
        )
