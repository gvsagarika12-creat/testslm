"""Behavioural tests for the vector store.

Every test runs against BOTH backends. The two implementations are meant to be
interchangeable, so any difference in behaviour is a bug in one of them.
"""
import pytest

from ragforge.chunk import Chunk
from ragforge.store import ChromaStore

DIM = 4


def make_chunk(index=0, doc_id="doc1", filename="a.pdf", text="hello world"):
    return Chunk(
        id=f"{doc_id}-{index}",
        doc_id=doc_id,
        text=text,
        source_filename=filename,
        page_start=1,
        page_end=1,
        chunk_index=index,
        token_count=2,
        chunk_size=400,
        overlap=60,
        ingested_at="2026-08-28T00:00:00+00:00",
    )


def unit(*values):
    """A length-DIM vector, L2-normalized."""
    padded = list(values) + [0.0] * (DIM - len(values))
    norm = sum(v * v for v in padded) ** 0.5 or 1.0
    return [v / norm for v in padded]


@pytest.fixture(params=["chroma", "postgres"])
def store(request, tmp_path):
    """An empty store. Parametrized so every test runs on both backends."""
    if request.param == "chroma":
        yield ChromaStore(persist_dir=tmp_path / "chroma", collection_name="test")
        return

    url = request.getfixturevalue("pg_database")("ragforge_test_store")
    from ragforge.pg_store import PgVectorStore

    pg = PgVectorStore(url, dimension=DIM)
    # Each test expects to start empty, matching the fresh tmp_path Chroma gets.
    with pg._conn.cursor() as cur:
        cur.execute("TRUNCATE chunks")
    try:
        yield pg
    finally:
        pg.close()


def test_starts_empty(store):
    assert store.count() == 0
    assert store.list_documents() == []


def test_upsert_then_count(store):
    store.upsert([make_chunk(0), make_chunk(1)], [unit(1, 0), unit(0, 1)])
    assert store.count() == 2


def test_query_returns_the_nearest_chunk_first(store):
    store.upsert(
        [make_chunk(0, text="cats"), make_chunk(1, text="ledgers")],
        [unit(1, 0), unit(0, 1)],
    )
    hits = store.query(unit(0.95, 0.05), k=2)
    assert hits[0].text == "cats"
    assert hits[0].score > hits[1].score


def test_scores_are_cosine_similarity_in_zero_to_one(store):
    store.upsert([make_chunk(0)], [unit(1, 0)])
    hit = store.query(unit(1, 0), k=1)[0]
    assert 0.99 <= hit.score <= 1.001


def test_hits_carry_provenance(store):
    store.upsert([make_chunk(0, filename="report.pdf")], [unit(1, 0)])
    hit = store.query(unit(1, 0), k=1)[0]
    assert hit.source_filename == "report.pdf"
    assert hit.page_start == 1
    assert hit.doc_id == "doc1"


def test_upserting_the_same_ids_replaces_rather_than_duplicates(store):
    store.upsert([make_chunk(0, text="first")], [unit(1, 0)])
    store.upsert([make_chunk(0, text="second")], [unit(1, 0)])
    assert store.count() == 1
    assert store.query(unit(1, 0), k=1)[0].text == "second"


def test_delete_by_filename_removes_only_that_document(store):
    store.upsert([make_chunk(0, doc_id="d1", filename="a.pdf")], [unit(1, 0)])
    store.upsert([make_chunk(0, doc_id="d2", filename="b.pdf")], [unit(0, 1)])
    removed = store.delete_by_filename("a.pdf")
    assert removed == 1
    assert store.count() == 1
    assert store.query(unit(0, 1), k=1)[0].source_filename == "b.pdf"


def test_get_document_params_reports_stored_settings(store):
    store.upsert([make_chunk(0, doc_id="d1")], [unit(1, 0)])
    assert store.get_document_params("d1") == {"chunk_size": 400, "overlap": 60}


def test_get_document_params_is_none_for_unknown_documents(store):
    assert store.get_document_params("nope") is None


def test_list_documents_groups_chunks(store):
    store.upsert(
        [make_chunk(0, doc_id="d1"), make_chunk(1, doc_id="d1")],
        [unit(1, 0), unit(0, 1)],
    )
    docs = store.list_documents()
    assert len(docs) == 1
    assert docs[0].chunk_count == 2
    assert docs[0].source_filename == "a.pdf"


def test_iter_chunks_round_trips_every_field(store):
    original = make_chunk(0)
    store.upsert([original], [unit(1, 0)])
    restored = list(store.iter_chunks())[0]
    assert restored == original


def test_iter_chunks_returns_documents_in_chunk_order(store):
    store.upsert(
        [make_chunk(1), make_chunk(0)],
        [unit(0, 1), unit(1, 0)],
    )
    assert [c.chunk_index for c in store.iter_chunks()] == [0, 1]


def test_persists_across_instances(tmp_path):
    path = tmp_path / "chroma"
    ChromaStore(path, "test").upsert([make_chunk(0)], [unit(1, 0)])
    assert ChromaStore(path, "test").count() == 1
