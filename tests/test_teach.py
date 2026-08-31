"""The 3H teaching layer, with a fake model so tests need no network."""
import json

import pytest

from ragforge.llm import LLMError, LLMResponse
from ragforge.store import Hit
from ragforge.teach import (
    TEACH_SCHEMA,
    Teacher,
    build_context,
    citation_label,
    load_system_prompt,
)

HITS = [
    Hit("c1", "d1", "Retinal artery occlusion is a blockage of the retinal artery.",
        "crao.pdf", 1, 1, 0.81),
    Hit("c2", "d1", "Treatments attempted include ocular massage and paracentesis.",
        "crao.pdf", 3, 4, 0.77),
]


def reply(**overrides):
    """A well-formed model reply, overridable per test."""
    base = {
        "topic": "Central retinal artery occlusion",
        "head": {"covered": True, "content": "A blockage of the retinal artery.",
                 "citations": ["crao.pdf p1"]},
        "heart": {"covered": False, "content": "", "citations": []},
        "hands": {"covered": True, "content": "Ocular massage is attempted.",
                  "citations": ["crao.pdf p3-4"]},
        "gap_report": ["heart: no consent or communication content in the sources"],
        "unverified_claims": [],
        "retrieval_question": "What determines outcome in CRAO?",
    }
    base.update(overrides)
    return base


class FakeClient:
    """Stands in for OllamaClient. Records what it was asked."""

    def __init__(self, payload=None, error=None):
        self.payload = payload if payload is not None else reply()
        self.error = error
        self.system = None
        self.user = None
        self.schema = None

    def chat_json(self, system, user, schema, num_ctx=16384):
        if self.error:
            raise self.error
        self.system, self.user, self.schema = system, user, schema
        return self.payload, LLMResponse(
            content=json.dumps(self.payload), thinking="",
            prompt_tokens=100, output_tokens=50, duration_seconds=5.0,
        )


class FakePipeline:
    def __init__(self, hits=None):
        self.hits = HITS if hits is None else hits
        self.query = None

    def search(self, query, k=5):
        self.query = query
        return self.hits[:k]


@pytest.fixture
def teacher():
    return Teacher(pipeline=FakePipeline(), client=FakeClient())


# --- citation labels --------------------------------------------------------

def test_single_page_label():
    assert citation_label(HITS[0]) == "crao.pdf p1"


def test_page_range_label():
    assert citation_label(HITS[1]) == "crao.pdf p3-4"


def test_context_prefixes_each_passage_with_its_label():
    context = build_context(HITS)
    assert "[crao.pdf p1]" in context
    assert "[crao.pdf p3-4]" in context
    assert "ocular massage" in context


# --- the happy path ---------------------------------------------------------

def test_answer_returns_all_three_vectors(teacher):
    answer = teacher.answer("How is CRAO treated?")
    assert set(answer.sections) == {"head", "heart", "hands"}


def test_covered_and_uncovered_are_reported(teacher):
    answer = teacher.answer("How is CRAO treated?")
    assert answer.covered_vectors == ["head", "hands"]
    assert answer.uncovered_vectors == ["heart"]


def test_gap_report_is_preserved(teacher):
    answer = teacher.answer("How is CRAO treated?")
    assert any("consent" in g for g in answer.gap_report)


def test_retrieved_hits_are_returned_for_inspection(teacher):
    answer = teacher.answer("How is CRAO treated?")
    assert len(answer.hits) == 2


def test_the_schema_is_sent_to_the_model():
    client = FakeClient()
    Teacher(pipeline=FakePipeline(), client=client).answer("q")
    assert client.schema == TEACH_SCHEMA


def test_the_prompt_carries_context_and_question():
    client = FakeClient()
    Teacher(pipeline=FakePipeline(), client=client).answer("How is CRAO treated?")
    assert "ocular massage" in client.user
    assert "How is CRAO treated?" in client.user


def test_the_system_prompt_is_the_3h_spec():
    client = FakeClient()
    Teacher(pipeline=FakePipeline(), client=client).answer("q")
    assert "3H PEDAGOGICAL AGENT" in client.system
    assert "MODE 3: TEACHING" in client.system


# --- citation verification (the safety net) ---------------------------------

def test_invented_citations_are_discarded():
    payload = reply(head={"covered": True, "content": "Something.",
                          "citations": ["crao.pdf p1", "made-up.pdf p9"]})
    answer = Teacher(FakePipeline(), FakeClient(payload)).answer("q")
    assert answer.sections["head"].citations == ["crao.pdf p1"]
    assert answer.sections["head"].dropped_citations == ["made-up.pdf p9"]
    assert any("made-up.pdf p9" in w for w in answer.warnings)


def test_bracketed_citations_are_accepted():
    """The model may or may not include brackets; both mean the same source."""
    payload = reply(head={"covered": True, "content": "X.",
                          "citations": ["[crao.pdf p1]"]})
    answer = Teacher(FakePipeline(), FakeClient(payload)).answer("q")
    assert answer.sections["head"].citations == ["crao.pdf p1"]
    assert answer.sections["head"].covered is True


def test_empty_string_citations_are_ignored():
    """gemma4 emits [""] for an empty section rather than []."""
    payload = reply(heart={"covered": False, "content": "", "citations": [""]})
    answer = Teacher(FakePipeline(), FakeClient(payload)).answer("q")
    assert answer.sections["heart"].citations == []
    assert answer.sections["heart"].dropped_citations == []


def test_a_covered_vector_with_no_valid_citation_is_demoted():
    """Content citing only invented sources is not grounded, whatever it claims."""
    payload = reply(heart={"covered": True,
                           "content": "Always obtain informed consent.",
                           "citations": ["invented.pdf p1"]})
    answer = Teacher(FakePipeline(), FakeClient(payload)).answer("q")
    assert answer.sections["heart"].covered is False
    assert any("no citation traceable" in w for w in answer.warnings)


def test_a_covered_vector_with_content_but_zero_citations_is_demoted():
    payload = reply(heart={"covered": True,
                           "content": "Consent must be obtained.", "citations": []})
    answer = Teacher(FakePipeline(), FakeClient(payload)).answer("q")
    assert answer.sections["heart"].covered is False


def test_an_honestly_empty_vector_produces_no_warning(teacher):
    answer = teacher.answer("How is CRAO treated?")
    assert answer.sections["heart"].covered is False
    assert answer.warnings == []


def test_unverified_claims_are_surfaced():
    payload = reply(unverified_claims=["Time to presentation affects outcome."])
    answer = Teacher(FakePipeline(), FakeClient(payload)).answer("q")
    assert answer.unverified_claims == ["Time to presentation affects outcome."]


# --- failure paths ----------------------------------------------------------

def test_empty_question_is_rejected(teacher):
    with pytest.raises(ValueError):
        teacher.answer("   ")


def test_no_retrieval_results_raises_rather_than_inventing():
    teacher = Teacher(FakePipeline(hits=[]), FakeClient())
    with pytest.raises(LLMError, match="nothing in the corpus"):
        teacher.answer("something absent")


def test_model_errors_propagate():
    teacher = Teacher(FakePipeline(), FakeClient(error=LLMError("server down")))
    with pytest.raises(LLMError, match="server down"):
        teacher.answer("q")


def test_missing_fields_do_not_crash():
    """A model may omit optional arrays; absence must not raise."""
    answer = Teacher(FakePipeline(), FakeClient({"topic": "T"})).answer("q")
    assert answer.gap_report == []
    assert answer.unverified_claims == []
    assert all(not s.covered for s in answer.sections.values())


# --- prompt loading ---------------------------------------------------------

def test_system_prompt_includes_both_files():
    prompt = load_system_prompt()
    assert "3H PEDAGOGICAL AGENT" in prompt      # 3h_agent.md
    assert "RAGForge deployment addendum" in prompt  # 3h_deployment.md


def test_missing_prompt_directory_is_a_clear_error(tmp_path):
    from ragforge.config import Settings

    config = Settings(data_dir=tmp_path / "d", prompts_dir=tmp_path / "nope")
    with pytest.raises(FileNotFoundError, match="missing prompt file"):
        load_system_prompt(config)
