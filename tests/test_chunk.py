import pytest

from ragforge.chunk import chunk_id, chunk_pages
from ragforge.parse import PageText

ISO = "2026-08-28T00:00:00+00:00"


def run(pages, tokenizer, chunk_size=10, overlap=3):
    return chunk_pages(
        pages,
        doc_id="doc",
        source_filename="f.pdf",
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        overlap=overlap,
        ingested_at=ISO,
    )


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_short_document_is_one_chunk(tokenizer):
    chunks = run([PageText(1, "alpha beta gamma")], tokenizer)
    assert len(chunks) == 1
    assert chunks[0].text == "alpha beta gamma"
    assert chunks[0].token_count == 3


def test_no_chunk_exceeds_chunk_size(tokenizer):
    chunks = run([PageText(1, words(95))], tokenizer, chunk_size=10, overlap=3)
    assert chunks, "expected chunks"
    assert all(c.token_count <= 10 for c in chunks)


def test_an_unbroken_run_longer_than_the_window_is_hard_split(tokenizer):
    # One "sentence" with no paragraph or sentence boundary anywhere.
    chunks = run([PageText(1, words(50))], tokenizer, chunk_size=10, overlap=0)
    assert all(c.token_count <= 10 for c in chunks)
    assert len(chunks) >= 5


def test_consecutive_chunks_overlap_but_do_not_exceed_the_budget(tokenizer):
    chunks = run([PageText(1, words(60))], tokenizer, chunk_size=10, overlap=3)
    assert len(chunks) >= 2
    for previous, current in zip(chunks, chunks[1:]):
        carried = set(previous.text.split()) & set(current.text.split())
        assert carried, "each chunk should re-include some of its predecessor"
        assert len(carried) <= 3


def test_zero_overlap_produces_disjoint_chunks(tokenizer):
    chunks = run([PageText(1, words(40))], tokenizer, chunk_size=10, overlap=0)
    for previous, current in zip(chunks, chunks[1:]):
        assert not (set(previous.text.split()) & set(current.text.split()))


def test_chunk_indexes_are_contiguous_from_zero(tokenizer):
    chunks = run([PageText(1, words(80))], tokenizer)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_page_provenance_for_a_chunk_spanning_two_pages(tokenizer):
    pages = [PageText(1, words(6, "a")), PageText(2, words(6, "b"))]
    chunks = run(pages, tokenizer, chunk_size=12, overlap=0)
    spanning = [c for c in chunks if c.page_start != c.page_end]
    assert spanning, "expected a chunk covering both pages"
    assert spanning[0].page_start == 1
    assert spanning[0].page_end == 2


def test_single_page_chunk_reports_that_page(tokenizer):
    chunks = run([PageText(7, "alpha beta")], tokenizer)
    assert chunks[0].page_start == 7
    assert chunks[0].page_end == 7


def test_paragraph_boundaries_are_preferred_over_mid_paragraph_splits(tokenizer):
    pages = [PageText(1, f"{words(8, 'p')}\n\n{words(8, 'q')}")]
    chunks = run(pages, tokenizer, chunk_size=10, overlap=0)
    # Neither chunk should mix the two paragraphs, since each fits alone.
    for c in chunks:
        tokens = set(c.text.split())
        assert not ({"p0"} & tokens and {"q0"} & tokens)


def test_parameters_are_recorded_on_every_chunk(tokenizer):
    chunks = run([PageText(1, words(30))], tokenizer, chunk_size=10, overlap=3)
    assert all(c.chunk_size == 10 and c.overlap == 3 for c in chunks)
    assert all(c.ingested_at == ISO for c in chunks)


def test_chunk_ids_are_deterministic_and_unique(tokenizer):
    first = run([PageText(1, words(40))], tokenizer)
    second = run([PageText(1, words(40))], tokenizer)
    assert [c.id for c in first] == [c.id for c in second]
    assert len({c.id for c in first}) == len(first)


def test_chunk_id_depends_on_document_and_index():
    assert chunk_id("a", 0) != chunk_id("b", 0)
    assert chunk_id("a", 0) != chunk_id("a", 1)
    assert len(chunk_id("a", 0)) == 32


def test_empty_input_yields_no_chunks(tokenizer):
    assert run([], tokenizer) == []


def test_overlap_must_be_smaller_than_chunk_size(tokenizer):
    with pytest.raises(ValueError):
        run([PageText(1, "a b c")], tokenizer, chunk_size=5, overlap=5)
