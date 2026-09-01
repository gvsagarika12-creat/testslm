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


def card(vector="head", headline="A claim.", bullets=None, citations=None):
    return {
        "vector": vector,
        "headline": headline,
        "bullets": bullets if bullets is not None else ["fact one", "fact two"],
        "citations": citations if citations is not None else ["crao.pdf p1"],
    }


def reply(**overrides):
    """A well-formed model reply, overridable per test."""
    base = {
        "title": "Management of Central Retinal Artery Occlusion",
        "overview": "CRAO is a blockage of the retinal artery causing sudden vision loss.",
        "cards": [
            card("head", "CRAO is an arterial blockage."),
            card("hands", "Ocular massage is attempted first.",
                 citations=["crao.pdf p3-4"]),
        ],
        "picture_this": "The retina pales while the fovea stays red.",
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


def teach(payload=None, hits=None):
    return Teacher(FakePipeline(hits), FakeClient(payload)).answer("How is CRAO treated?")


@pytest.fixture
def answer():
    return teach()


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


# --- the answer shape -------------------------------------------------------

def test_title_and_overview_come_first(answer):
    assert answer.title.startswith("Management of")
    assert "sudden vision loss" in answer.overview


def test_cards_carry_headline_and_bullets(answer):
    assert answer.cards[0].headline == "CRAO is an arterial blockage."
    assert answer.cards[0].bullets == ["fact one", "fact two"]


def test_picture_this_is_captured(answer):
    assert "fovea" in answer.picture_this


def test_retrieval_question_is_captured(answer):
    assert answer.retrieval_question == "What determines outcome in CRAO?"


def test_cards_are_ordered_head_then_heart_then_hands():
    payload = reply(cards=[
        card("hands", "H."), card("heart", "E.", citations=["crao.pdf p1"]),
        card("head", "A."),
    ])
    assert [c.vector for c in teach(payload).cards] == ["head", "heart", "hands"]


def test_several_cards_per_vector_are_kept():
    payload = reply(cards=[
        card("hands", "Medical management."),
        card("hands", "Surgical escalation."),
    ])
    assert len(teach(payload).cards_for("hands")) == 2


def test_sources_are_deduplicated_in_order(answer):
    assert answer.sources == ["crao.pdf p1", "crao.pdf p3-4"]


def test_coverage_is_derived_from_grounded_cards(answer):
    assert answer.covered_vectors == ["head", "hands"]
    assert answer.uncovered_vectors == ["heart"]


def test_gap_report_is_preserved(answer):
    assert any("consent" in g for g in answer.gap_report)


def test_retrieved_hits_are_returned_for_inspection(answer):
    assert len(answer.hits) == 2


# --- what is sent to the model ----------------------------------------------

def test_the_schema_is_sent_to_the_model():
    client = FakeClient()
    Teacher(FakePipeline(), client).answer("q")
    assert client.schema == TEACH_SCHEMA


def test_the_prompt_carries_context_and_question():
    client = FakeClient()
    Teacher(FakePipeline(), client).answer("How is CRAO treated?")
    assert "ocular massage" in client.user
    assert "How is CRAO treated?" in client.user


def test_the_system_prompt_is_the_3h_spec():
    client = FakeClient()
    Teacher(FakePipeline(), client).answer("q")
    assert "3H PEDAGOGICAL AGENT" in client.system
    assert "MODE 3: TEACHING" in client.system


# --- citation verification (the safety net) ---------------------------------

def test_invented_citations_are_discarded():
    payload = reply(cards=[card(citations=["crao.pdf p1", "made-up.pdf p9"])])
    result = teach(payload)
    assert result.cards[0].citations == ["crao.pdf p1"]
    assert result.cards[0].dropped_citations == ["made-up.pdf p9"]
    assert any("made-up.pdf p9" in w for w in result.warnings)


def test_bracketed_citations_are_accepted():
    """The model may or may not include brackets; both mean the same source."""
    result = teach(reply(cards=[card(citations=["[crao.pdf p1]"])]))
    assert result.cards[0].citations == ["crao.pdf p1"]
    assert result.cards[0].grounded


def test_empty_string_citations_are_ignored():
    """gemma4 emits [""] rather than [] in some replies."""
    result = teach(reply(cards=[card(citations=[""])]))
    assert result.cards[0].citations == []
    assert result.cards[0].dropped_citations == []


def test_a_card_citing_only_invented_sources_is_flagged_ungrounded():
    result = teach(reply(cards=[card(citations=["invented.pdf p1"])]))
    assert result.cards[0].grounded is False
    assert any("no citation traceable" in w for w in result.warnings)


def test_an_ungrounded_card_does_not_count_towards_coverage():
    result = teach(reply(cards=[card("heart", "Consent matters.", citations=[])]))
    assert "heart" not in result.covered_vectors


def test_a_clean_answer_produces_no_warnings(answer):
    assert answer.warnings == []


def test_unverified_claims_are_surfaced():
    result = teach(reply(unverified_claims=["Time to presentation affects outcome."]))
    assert result.unverified_claims == ["Time to presentation affects outcome."]


# --- malformed model output -------------------------------------------------

def test_cards_with_an_unknown_vector_are_dropped():
    result = teach(reply(cards=[card("spleen", "Nonsense."), card("head", "Real.")]))
    assert [c.vector for c in result.cards] == ["head"]


def test_cards_without_a_headline_are_dropped():
    result = teach(reply(cards=[card(headline="   "), card(headline="Real.")]))
    assert len(result.cards) == 1


def test_non_dict_cards_do_not_crash():
    result = teach(reply(cards=["not a card", None, card()]))
    assert len(result.cards) == 1


def test_missing_fields_do_not_crash():
    result = teach({"title": "T"})
    assert result.cards == []
    assert result.gap_report == []
    assert result.overview == ""
    assert result.uncovered_vectors == ["head", "heart", "hands"]


# --- failure paths ----------------------------------------------------------

def test_empty_question_is_rejected():
    with pytest.raises(ValueError):
        Teacher(FakePipeline(), FakeClient()).answer("   ")


def test_no_retrieval_results_raises_rather_than_inventing():
    with pytest.raises(LLMError, match="nothing in the corpus"):
        Teacher(FakePipeline(hits=[]), FakeClient()).answer("something absent")


def test_model_errors_propagate():
    teacher = Teacher(FakePipeline(), FakeClient(error=LLMError("server down")))
    with pytest.raises(LLMError, match="server down"):
        teacher.answer("q")


# --- prompt loading ---------------------------------------------------------

def test_system_prompt_includes_both_files():
    prompt = load_system_prompt()
    assert "3H PEDAGOGICAL AGENT" in prompt          # 3h_agent.md
    assert "RAGForge deployment addendum" in prompt  # 3h_deployment.md


def test_deployment_prompt_describes_the_card_shape():
    prompt = load_system_prompt()
    assert "picture_this" in prompt
    assert "headline" in prompt


def test_missing_prompt_directory_is_a_clear_error(tmp_path):
    from ragforge.config import Settings

    config = Settings(data_dir=tmp_path / "d", prompts_dir=tmp_path / "nope")
    with pytest.raises(FileNotFoundError, match="missing prompt file"):
        load_system_prompt(config)
