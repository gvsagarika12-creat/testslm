"""Full-stack tests using the real embedding model and a real Chroma database.

Run with: pytest -m slow
"""
import pytest

from ragforge.config import Settings
from ragforge.embed import Embedder
from ragforge.export import export_jsonl
from ragforge.pipeline import Pipeline
from ragforge.store import ChromaStore

pytestmark = pytest.mark.slow

PAGE_ONE = (
    "The quarterly financial report covers revenue, operating margin, and "
    "headcount changes across all business units for the period."
)
PAGE_TWO = (
    "Migratory bird populations along the northern flyway were surveyed using "
    "banding data collected over twelve consecutive breeding seasons."
)


@pytest.fixture
def real_pipeline(tmp_path):
    config = Settings(data_dir=tmp_path / "data")
    config.ensure_dirs()
    return Pipeline(
        store=ChromaStore(config.chroma_dir, config.collection_name),
        embedder=Embedder(config),
        config=config,
    )


def test_ingest_then_search_returns_the_right_page(real_pipeline, make_pdf):
    pdf = make_pdf([PAGE_ONE, PAGE_TWO], name="mixed.pdf")
    result = real_pipeline.ingest_file(pdf, chunk_size=32, overlap=8)
    assert result.status == "ingested"

    hits = real_pipeline.search("how many birds were counted in the survey", k=2)
    assert hits, "expected at least one hit"
    assert "bird" in hits[0].text.lower()

    hits = real_pipeline.search("what was the operating margin", k=2)
    assert "margin" in hits[0].text.lower()


def test_reingesting_the_same_file_does_not_grow_the_store(real_pipeline, make_pdf):
    pdf = make_pdf([PAGE_ONE], name="stable.pdf")
    real_pipeline.ingest_file(pdf)
    count_after_first = real_pipeline.store.count()
    real_pipeline.ingest_file(pdf)
    assert real_pipeline.store.count() == count_after_first


def test_forced_reingestion_keeps_the_count_stable(real_pipeline, make_pdf):
    pdf = make_pdf([PAGE_ONE], name="stable.pdf")
    real_pipeline.ingest_file(pdf)
    count_after_first = real_pipeline.store.count()
    real_pipeline.ingest_file(pdf, force=True)
    assert real_pipeline.store.count() == count_after_first


def test_every_stored_chunk_fits_the_model_window(real_pipeline, make_pdf):
    body = " ".join([PAGE_ONE, PAGE_TWO] * 20)
    real_pipeline.ingest_file(make_pdf([body], name="long.pdf"))
    tokenizer = real_pipeline.embedder.tokenizer
    limit = real_pipeline.config.max_model_tokens
    for chunk in real_pipeline.store.iter_chunks():
        assert tokenizer.count_tokens(chunk.text) <= limit


def test_export_covers_every_stored_chunk(real_pipeline, make_pdf, tmp_path):
    real_pipeline.ingest_file(make_pdf([PAGE_ONE, PAGE_TWO], name="mixed.pdf"))
    out = tmp_path / "corpus.jsonl"
    assert export_jsonl(real_pipeline.store, out) == real_pipeline.store.count()
