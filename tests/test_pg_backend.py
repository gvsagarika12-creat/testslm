"""The Postgres backend wired through the full pipeline, and backend selection."""
import pytest

from ragforge.config import Settings
from ragforge.pipeline import Pipeline, build_store
from tests.test_pipeline import FakeEmbedder

DIM = 8


@pytest.fixture
def pg_pipeline(tmp_path, pg_database):
    from ragforge.pg_store import PgVectorStore

    url = pg_database("ragforge_test_pipeline")
    config = Settings(
        data_dir=tmp_path / "data",
        chunk_size=20,
        chunk_overlap=5,
        store_backend="postgres",
        database_url=url,
        embedding_dimension=DIM,
    )
    config.ensure_dirs()
    store = PgVectorStore(url, dimension=DIM)
    with store._conn.cursor() as cur:
        cur.execute("TRUNCATE chunks")
    try:
        yield Pipeline(store=store, embedder=FakeEmbedder(), config=config)
    finally:
        store.close()


def test_ingest_stores_chunks_in_postgres(pg_pipeline, make_pdf):
    result = pg_pipeline.ingest_file(make_pdf(["alpha beta gamma delta"]))
    assert result.status == "ingested"
    assert pg_pipeline.store.count() == result.chunk_count


def test_search_round_trips_through_postgres(pg_pipeline, make_pdf):
    pg_pipeline.ingest_file(make_pdf(["quarterly revenue figures"], name="fin.pdf"))
    pg_pipeline.ingest_file(make_pdf(["migratory bird patterns"], name="bio.pdf"))
    hits = pg_pipeline.search("quarterly revenue figures", k=2)
    assert hits[0].source_filename == "fin.pdf"


def test_reingest_with_same_params_is_skipped(pg_pipeline, make_pdf):
    pdf = make_pdf(["alpha beta gamma"])
    first = pg_pipeline.ingest_file(pdf)
    second = pg_pipeline.ingest_file(pdf)
    assert second.status == "skipped"
    assert pg_pipeline.store.count() == first.chunk_count


def test_reingest_with_new_chunk_size_replaces_not_duplicates(pg_pipeline, make_pdf):
    pdf = make_pdf([" ".join(f"w{i}" for i in range(60))], name="doc.pdf")
    pg_pipeline.ingest_file(pdf, chunk_size=10, overlap=0)
    pg_pipeline.ingest_file(pdf, chunk_size=20, overlap=0)
    assert len({c.chunk_size for c in pg_pipeline.store.iter_chunks()}) == 1


def test_stats_reports_the_postgres_backend(pg_pipeline, make_pdf):
    pg_pipeline.ingest_file(make_pdf(["alpha beta gamma"]))
    stats = pg_pipeline.stats()
    assert stats["backend"] == "postgres"
    assert stats["documents"] == 1
    assert "ragforge" in stats["location"]


def test_stats_location_never_leaks_the_password(pg_pipeline, make_pdf):
    pg_pipeline.ingest_file(make_pdf(["alpha beta"]))
    assert "ragforge:ragforge@" not in pg_pipeline.stats()["location"]


def test_build_store_selects_postgres(tmp_path, pg_database):
    from ragforge.pg_store import PgVectorStore

    config = Settings(
        data_dir=tmp_path / "d",
        store_backend="postgres",
        database_url=pg_database("ragforge_test_pipeline"),
        embedding_dimension=DIM,
    )
    store = build_store(config)
    assert isinstance(store, PgVectorStore)
    store.close()


def test_build_store_still_supports_chroma(tmp_path):
    from ragforge.store import ChromaStore

    config = Settings(data_dir=tmp_path / "d", store_backend="chroma")
    config.ensure_dirs()
    assert isinstance(build_store(config), ChromaStore)


def test_postgres_is_the_default_backend():
    assert Settings().store_backend == "postgres"
