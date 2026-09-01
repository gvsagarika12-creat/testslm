"""The 3H layer end to end: real corpus, real retrieval, real model.

Run with: pytest -m slow tests/test_teach_live.py
"""
import pytest

from ragforge.pipeline import build_pipeline
from ragforge.teach import VECTORS, Teacher, citation_label

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def teacher():
    pipeline = build_pipeline()
    if pipeline.store.count() == 0:
        pytest.skip("corpus is empty — ingest documents first")
    return Teacher(pipeline=pipeline)


@pytest.fixture(scope="module")
def answer(teacher):
    return teacher.answer("How is central retinal artery occlusion treated?", k=6)


# --- the shape the interface renders ----------------------------------------

def test_it_produces_a_title_and_an_overview(answer):
    assert answer.title.strip()
    assert len(answer.overview.split()) >= 15, "overview should be a paragraph"


def test_it_produces_teaching_cards(answer):
    assert answer.cards, "no teaching cards were produced"


def test_every_card_has_a_headline_and_bullets(answer):
    for c in answer.cards:
        assert c.headline.strip()
        assert c.bullets, f"card '{c.headline[:40]}' has no supporting bullets"


def test_it_produces_a_dual_coding_cue(answer):
    """§7 MODE 3 requires a visual/spatial cue for the learner to picture."""
    assert answer.picture_this.strip()


def test_it_produces_a_retrieval_question(answer):
    assert answer.retrieval_question.strip()


def test_cards_are_taught_in_head_heart_hands_order(answer):
    order = [VECTORS.index(c.vector) for c in answer.cards]
    assert order == sorted(order)


# --- grounding --------------------------------------------------------------

def test_every_citation_is_traceable_to_a_retrieved_passage(answer):
    allowed = {citation_label(h) for h in answer.hits}
    for c in answer.cards:
        for citation in c.citations:
            assert citation in allowed, f"untraceable citation: {citation}"


def test_no_citations_were_discarded_as_invented(answer):
    """A real failure here means the model fabricated sources."""
    invented = {c.headline[:40]: c.dropped_citations for c in answer.cards if c.dropped_citations}
    assert not invented, f"model invented citations: {invented}"


def test_at_least_one_vector_is_covered(answer):
    assert answer.covered_vectors, "no card traced to any retrieved passage"


def test_an_uncovered_vector_produces_a_gap_report_line(answer):
    """The invariant, not the judgement.

    Whether the model rates a vector covered on a given run is its discretion.
    What must always hold is that leaving one uncovered is explained.
    """
    if answer.uncovered_vectors:
        assert answer.gap_report, (
            f"{answer.uncovered_vectors} uncovered but no gap report given"
        )


def test_citations_do_not_leak_into_the_prose(answer):
    """D5: the interface renders citations; the prose must read cleanly."""
    filenames = {h.source_filename for h in answer.hits}
    prose = " ".join(
        [answer.overview, answer.picture_this]
        + [c.headline for c in answer.cards]
        + [b for c in answer.cards for b in c.bullets]
    )
    leaked = [f for f in filenames if f in prose]
    assert not leaked, f"citation labels leaked into prose: {leaked}"


def test_vectors_respect_the_350_word_budget(answer):
    """§9: teaching chunk <= 350 words, counted across a vector's cards."""
    for vector in VECTORS:
        words = sum(
            len(c.headline.split()) + sum(len(b.split()) for b in c.bullets)
            for c in answer.cards_for(vector)
        )
        assert words <= 350, f"{vector} overran the budget at {words} words"


def test_an_unanswerable_question_still_stays_grounded(teacher):
    """Nothing in the corpus is about bread. It must not invent sources."""
    result = teacher.answer("What temperature should I bake sourdough bread at?", k=4)
    allowed = {citation_label(h) for h in result.hits}
    for c in result.cards:
        for citation in c.citations:
            assert citation in allowed
