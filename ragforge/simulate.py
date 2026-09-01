"""Case simulation — the only mode that can measure HEART.

A factual question exercises HEAD alone: there is nothing in "what is the
purpose of laser photocoagulation" for §8.3's BARS scale to score. Empathy is
behavioural, so it has to be elicited before it can be measured.

A case does that. The patient speaks; the learner must reply to them, in their
own words; those words are the evidence §8.3 scores. Note that this works even
though the corpus contains no HEART content — the passages supply the clinical
facts, the learner supplies the empathy. The corpus gap blocked *teaching*
HEART, never *assessing* it.

Three scenes, one per vector, then a scored debrief (§7 MODE 6 → MODE 4 → 5).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ragforge.config import Settings, settings as default_settings
from ragforge.llm import LLMError, LLMResponse, OllamaClient
from ragforge.prompts import build_system_prompt
from ragforge.store import Hit
from ragforge.teach import (
    ASSESS_SCHEMA,
    VECTORS,
    Assessment,
    _normalize_citation,
    build_context,
    citation_label,
    parse_assessment,
)

SCENE_ORDER = VECTORS  # head, then heart, then hands

_SCENE_SCHEMA = {
    "type": "object",
    "properties": {
        "vector": {"type": "string", "enum": list(VECTORS)},
        "situation": {"type": "string"},
        "prompt": {"type": "string"},
    },
    "required": ["vector", "situation", "prompt"],
}

START_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "presentation": {"type": "string"},
        "persona": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
        "scene": _SCENE_SCHEMA,
    },
    "required": ["title", "presentation", "persona", "citations", "scene"],
}

RESPOND_SCHEMA = {
    "type": "object",
    "properties": {
        "reaction": {"type": "string"},
        "scene": _SCENE_SCHEMA,
    },
    "required": ["reaction"],
}


@dataclass
class Scene:
    vector: str
    situation: str
    prompt: str

    @property
    def index(self) -> int:
        return SCENE_ORDER.index(self.vector)


@dataclass
class Turn:
    """One completed exchange: what was asked, what the learner said, what happened."""

    scene: Scene
    learner_reply: str
    reaction: str = ""


@dataclass
class CaseSession:
    title: str
    presentation: str
    persona: str
    citations: list[str]
    hits: list[Hit]
    scene: Scene | None
    turns: list[Turn] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    debrief: Assessment | None = None

    @property
    def finished(self) -> bool:
        return self.scene is None

    @property
    def scene_number(self) -> int:
        return len(self.turns) + 1

    def transcript(self) -> str:
        """The case so far, as the model will re-read it on the next turn."""
        lines = [f"CASE: {self.title}", self.presentation, f"PATIENT: {self.persona}"]
        for turn in self.turns:
            lines.append(f"\n[SCENE {turn.scene.index + 1} — {turn.scene.vector.upper()}]")
            lines.append(turn.scene.situation)
            lines.append(f"ASKED: {turn.scene.prompt}")
            lines.append(f"LEARNER SAID: {turn.learner_reply}")
            if turn.reaction:
                lines.append(f"WHAT HAPPENED: {turn.reaction}")
        return "\n".join(lines)


def _parse_scene(raw, expected_vector: str | None) -> Scene | None:
    if not isinstance(raw, dict):
        return None
    situation = (raw.get("situation") or "").strip()
    prompt = (raw.get("prompt") or "").strip()
    if not prompt:
        return None
    vector = str(raw.get("vector") or "").strip().lower()
    # The scene order is fixed by design; trust the sequence, not the model.
    if expected_vector:
        vector = expected_vector
    elif vector not in VECTORS:
        return None
    return Scene(vector=vector, situation=situation, prompt=prompt)


class Simulator:
    """Runs a three-scene case and scores it."""

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

    def _system(self) -> str:
        return build_system_prompt("simulate", self.config)

    def start(
        self,
        topic: str,
        k: int | None = None,
        hits: Sequence[Hit] | None = None,
    ) -> CaseSession:
        """Build a case from the corpus and open scene 1.

        Pass `hits` to build the case from passages already retrieved — when a
        case follows a teaching answer, it must be built from the same material
        the learner was just taught. §3's misalignment check: assess what was
        actually taught, not something adjacent the retriever happened to find.
        """
        if not topic.strip():
            raise ValueError("a topic is required")

        if hits is None:
            hits = self.pipeline.search(topic, k=k or self.config.teach_context_chunks)
        if not hits:
            raise LLMError(
                "nothing in the corpus matches this topic — ingest relevant "
                "documents first"
            )

        allowed = {citation_label(h) for h in hits}
        parsed, _ = self.client.chat_json(
            system=self._system(),
            user=(
                f"CONTEXT:\n{build_context(hits)}\n\n"
                f"SIMULATION STEP: start\n"
                f"TOPIC: {topic.strip()}\n\n"
                "Build the case and open scene 1 (head)."
            ),
            schema=START_SCHEMA,
        )

        kept, dropped = [], []
        for citation in parsed.get("citations") or []:
            normalized = _normalize_citation(str(citation))
            if not normalized:
                continue
            (kept if normalized in allowed else dropped).append(normalized)

        scene = _parse_scene(parsed.get("scene"), SCENE_ORDER[0])
        if scene is None:
            raise LLMError("the model did not produce an opening scene")

        warnings = []
        if dropped:
            warnings.append(
                "case citations not in the retrieved passages — " + ", ".join(dropped)
            )

        return CaseSession(
            title=(parsed.get("title") or topic).strip(),
            presentation=(parsed.get("presentation") or "").strip(),
            persona=(parsed.get("persona") or "").strip(),
            citations=kept,
            hits=list(hits),
            scene=scene,
            warnings=warnings,
        )

    def respond(self, session: CaseSession, learner_reply: str) -> CaseSession:
        """Record the learner's reply, react in character, and open the next scene."""
        if session.scene is None:
            raise ValueError("the case has already finished")
        if not learner_reply.strip():
            raise ValueError("a reply is required")

        current = session.scene
        next_index = current.index + 1
        next_vector = (
            SCENE_ORDER[next_index] if next_index < len(SCENE_ORDER) else None
        )

        instruction = (
            f"Open scene {next_index + 1} ({next_vector})."
            if next_vector
            else "This was the final scene. Return the reaction only; set scene to null."
        )
        parsed, _ = self.client.chat_json(
            system=self._system(),
            user=(
                f"CONTEXT:\n{build_context(session.hits)}\n\n"
                f"SIMULATION STEP: respond\n\n"
                f"THE CASE SO FAR:\n{session.transcript()}\n\n"
                f"[SCENE {current.index + 1} — {current.vector.upper()}]\n"
                f"{current.situation}\n"
                f"ASKED: {current.prompt}\n\n"
                f"THE LEARNER'S REPLY:\n{learner_reply.strip()}\n\n{instruction}"
            ),
            schema=RESPOND_SCHEMA,
        )

        reaction = (parsed.get("reaction") or "").strip()
        session.turns.append(
            Turn(scene=current, learner_reply=learner_reply.strip(), reaction=reaction)
        )
        session.scene = (
            _parse_scene(parsed.get("scene"), next_vector) if next_vector else None
        )
        return session

    def score(self, session: CaseSession) -> Assessment:
        """MODE 4 → MODE 5 over the whole case. All three vectors are assessed."""
        if not session.turns:
            raise ValueError("the case has no completed scenes to score")

        parsed, meta = self.client.chat_json(
            system=self._system(),
            user=(
                f"CONTEXT:\n{build_context(session.hits)}\n\n"
                f"SIMULATION STEP: debrief\n\n"
                f"THE COMPLETED CASE:\n{session.transcript()}\n\n"
                "Score all three vectors against the §8 anchors using the "
                "learner's own words as evidence, then give MODE 5 feedback."
            ),
            schema=ASSESS_SCHEMA,
        )

        assessment = parse_assessment(
            parsed,
            question=f"Case: {session.title}",
            learner_answer="\n\n".join(t.learner_reply for t in session.turns),
            allowed={citation_label(h) for h in session.hits},
            meta=meta,
        )
        session.debrief = assessment
        return assessment
