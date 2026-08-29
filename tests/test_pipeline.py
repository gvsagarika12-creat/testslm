import pytest

from ragforge.config import Settings
from ragforge.pipeline import Pipeline
from ragforge.store import ChromaStore
from tests.conftest import WhitespaceTokenizer


class FakeEmbedder:
    """Deterministic vectors, no model download. Dimension 8."""

    def __init__(self):
        self.tokenizer = WhitespaceTokenizer()
        self.calls = 0

    def _vector(self, text):
        vector = [0.0] * 8
        for word in text.split():
            # Stable across processes, unlike hash().
            bucket = sum(ord(c) for c in word) % 8
            vector[bucket] += 1.0
        norm = sum(v * v for v in vector) ** 0.5 or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts, batch_size=None):
        self.calls += 1
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


@pytest.fixture
def pipeline(tmp_path):
    config = Settings(data_dir=tmp_path / "data", chunk_size=20, chunk_overlap=5)
    config.ensure_dirs()
    store = ChromaStore(config.chroma_dir, config.collection_name)
    return Pipeline(store=store, embedder=FakeEmbedder(), config=config)


def test_ingesting_a_pdf_reports_success_and_stores_chunks(pipeline, make_pdf):
    path = make_pdf(["alpha beta gamma delta", "epsilon zeta"])
    result = pipeline.ingest_file(path)
    assert result.status == "ingested"
    assert result.chunk_count > 0
    assert pipeline.store.count() == result.chunk_count


def test_the_source_pdf_is_copied_into_uploads(pipeline, make_pdf):
    path = make_pdf(["alpha beta"], name="report.pdf")
    pipeline.ingest_file(path)
    assert (pipeline.config.uploads_dir / "report.pdf").is_file()


def test_reingesting_identical_bytes_with_identical_params_is_skipped(pipeline, make_pdf):
    path = make_pdf(["alpha beta gamma"])
    first = pipeline.ingest_file(path)
    second = pipeline.ingest_file(path)
    assert second.status == "skipped"
    assert "already ingested" in second.message
    assert pipeline.store.count() == first.chunk_count


def test_reingesting_with_different_chunk_size_reprocesses(pipeline, make_pdf):
    path = make_pdf([" ".join(f"w{i}" for i in range(60))])
    pipeline.ingest_file(path, chunk_size=20, overlap=0)
    coarse = pipeline.store.count()
    result = pipeline.ingest_file(path, chunk_size=10, overlap=0)
    assert result.status == "ingested"
    assert pipeline.store.count() != coarse
    assert all(c.chunk_size == 10 for c in pipeline.store.iter_chunks())


def test_force_reingests_even_with_identical_params(pipeline, make_pdf):
    path = make_pdf(["alpha beta gamma"])
    pipeline.ingest_file(path)
    assert pipeline.ingest_file(path, force=True).status == "ingested"


def test_reingestion_leaves_no_orphan_chunks_from_the_old_version(pipeline, make_pdf):
    path = make_pdf([" ".join(f"w{i}" for i in range(60))], name="doc.pdf")
    pipeline.ingest_file(path, chunk_size=10, overlap=0)
    pipeline.ingest_file(path, chunk_size=20, overlap=0)
    assert len({c.chunk_size for c in pipeline.store.iter_chunks()}) == 1


def test_encrypted_pdf_fails_without_raising(pipeline, make_pdf):
    result = pipeline.ingest_file(make_pdf(["secret"], encrypt=True))
    assert result.status == "failed"
    assert "password" in result.message.lower()
    assert pipeline.store.count() == 0


def test_pdf_without_a_text_layer_fails_with_the_ocr_hint(pipeline, make_pdf):
    result = pipeline.ingest_file(make_pdf(["", ""]))
    assert result.status == "failed"
    assert "ocr" in result.message.lower()


def test_corrupt_file_fails_without_raising(pipeline, tmp_path):
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"not a pdf at all")
    assert pipeline.ingest_file(bad).status == "failed"


def test_nothing_is_written_when_a_file_fails(pipeline, make_pdf):
    pipeline.ingest_file(make_pdf(["good text here"]))
    before = pipeline.store.count()
    pipeline.ingest_file(make_pdf(["", ""]))
    assert pipeline.store.count() == before


def test_a_bad_file_does_not_abort_the_batch(pipeline, make_pdf, tmp_path):
    make_pdf(["alpha beta"], name="batch/ok1.pdf")
    make_pdf(["gamma delta"], name="batch/ok2.pdf")
    folder = tmp_path / "batch"
    (folder / "broken.pdf").write_bytes(b"garbage")

    report = pipeline.ingest_path(folder)
    assert len(report.ingested) == 2
    assert len(report.failed) == 1
    assert report.total_chunks > 0


def test_ingest_path_ignores_non_pdf_files(pipeline, tmp_path, make_pdf):
    make_pdf(["alpha beta"], name="mixed/doc.pdf")
    folder = tmp_path / "mixed"
    (folder / "notes.txt").write_text("ignore me")
    report = pipeline.ingest_path(folder)
    assert len(report.results) == 1


def test_ingest_path_accepts_a_single_file(pipeline, make_pdf):
    report = pipeline.ingest_path(make_pdf(["alpha beta"]))
    assert len(report.results) == 1


def test_ingest_path_recurses_when_asked(pipeline, tmp_path, make_pdf):
    make_pdf(["alpha beta"], name="top/inner/deep.pdf")
    assert len(pipeline.ingest_path(tmp_path / "top", recursive=True).results) == 1
    assert len(pipeline.ingest_path(tmp_path / "top", recursive=False).results) == 0


def test_search_finds_the_relevant_chunk(pipeline, make_pdf):
    pipeline.ingest_file(make_pdf(["quarterly revenue figures"], name="fin.pdf"))
    pipeline.ingest_file(make_pdf(["migratory bird patterns"], name="bio.pdf"))
    hits = pipeline.search("quarterly revenue figures", k=2)
    assert hits[0].source_filename == "fin.pdf"


def test_search_on_an_empty_store_returns_nothing(pipeline):
    assert pipeline.search("anything") == []


def test_stats_reports_documents_and_chunks(pipeline, make_pdf):
    pipeline.ingest_file(make_pdf(["alpha beta gamma"]))
    stats = pipeline.stats()
    assert stats["documents"] == 1
    assert stats["chunks"] == pipeline.store.count()
