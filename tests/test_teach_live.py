"""The 3H layer end to end: real corpus, real retrieval, real model.

Run with: pytest -m slow tests/test_teach_live.py
"""
import pytest

from ragforge.llm import LLMError
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


def test_all_three_vectors_are_reported(answer):
    assert set(answer.sections) == set(VECTORS)


def test_at_least_one_vector_is_covered(answer):
    assert answer.covered_vectors, "the model found nothing usable in the corpus"


def test_every_citation_is_traceable_to_a_retrieved_passage(answer):
    allowed = {citation_label(h) for h in answer.hits}
    for section in answer.sections.values():
        for citation in section.citations:
            assert citation in allowed, f"untraceable citation: {citation}"


def test_no_citations_were_discarded_as_invented(answer):
    """A real failure here means the model fabricated sources."""
    invented = {
        s.vector: s.dropped_citations
        for s in answer.sections.values()
        if s.dropped_citations
    }
    assert not invented, f"model invented citations: {invented}"


def test_covered_sections_carry_at_least_one_citation(answer):
    for section in answer.sections.values():
        if section.covered:
            assert section.citations, f"{section.vector} is covered but uncited"


def test_uncovered_sections_are_empty(answer):
    for section in answer.sections.values():
        if not section.covered:
            assert not section.content.strip()


def test_it_produces_a_retrieval_question(answer):
    assert answer.retrieval_question.strip()


def test_sections_respect_the_350_word_budget(answer):
    """§9 of the 3H spec: teaching chunk <= 350 words."""
    for section in answer.sections.values():
        assert len(section.content.split()) <= 350, f"{section.vector} overran"


def test_an_uncovered_vector_produces_a_gap_report_line(answer):
    """The invariant, not the judgement.

    Whether the model rates HEART covered on a given run varies — that is its
    discretion, not a contract term. What must always hold is that declaring a
    vector uncovered is accompanied by an explanation of what is missing.
    """
    if answer.uncovered_vectors:
        assert answer.gap_report, (
            f"{answer.uncovered_vectors} uncovered but no gap report given"
        )


def test_an_unanswerable_question_still_stays_grounded(teacher):
    """Nothing in the corpus is about bread. It must not invent sources."""
    result = teacher.answer("What temperature should I bake sourdough bread at?", k=4)
    allowed = {citation_label(h) for h in result.hits}
    for section in result.sections.values():
        for citation in section.citations:
            assert citation in allowed
