from typer.testing import CliRunner

import cli as cli_module
from ragforge.config import Settings
from ragforge.pipeline import Pipeline
from ragforge.store import ChromaStore
from tests.test_pipeline import FakeEmbedder

runner = CliRunner()


def _install_test_pipeline(monkeypatch, tmp_path):
    config = Settings(data_dir=tmp_path / "data", chunk_size=20, chunk_overlap=5)
    config.ensure_dirs()
    pipeline = Pipeline(
        store=ChromaStore(config.chroma_dir, config.collection_name),
        embedder=FakeEmbedder(),
        config=config,
    )
    monkeypatch.setattr(cli_module, "build_pipeline", lambda config=None: pipeline)
    return pipeline


def test_ingest_reports_per_file_status(monkeypatch, tmp_path, make_pdf):
    _install_test_pipeline(monkeypatch, tmp_path)
    pdf = make_pdf(["alpha beta gamma"], name="doc.pdf")
    result = runner.invoke(cli_module.app, ["ingest", str(pdf)])
    assert result.exit_code == 0
    assert "ingested" in result.stdout
    assert "doc.pdf" in result.stdout


def test_ingest_exits_nonzero_when_every_file_fails(monkeypatch, tmp_path):
    _install_test_pipeline(monkeypatch, tmp_path)
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"garbage")
    result = runner.invoke(cli_module.app, ["ingest", str(bad)])
    assert result.exit_code == 1
    assert "failed" in result.stdout


def test_search_prints_ranked_hits(monkeypatch, tmp_path, make_pdf):
    pipeline = _install_test_pipeline(monkeypatch, tmp_path)
    pipeline.ingest_file(make_pdf(["quarterly revenue figures"], name="fin.pdf"))
    result = runner.invoke(cli_module.app, ["search", "quarterly revenue figures"])
    assert result.exit_code == 0
    assert "fin.pdf" in result.stdout


def test_stats_prints_counts(monkeypatch, tmp_path, make_pdf):
    pipeline = _install_test_pipeline(monkeypatch, tmp_path)
    pipeline.ingest_file(make_pdf(["alpha beta"]))
    result = runner.invoke(cli_module.app, ["stats"])
    assert result.exit_code == 0
    assert "documents" in result.stdout.lower()


def test_export_writes_the_file(monkeypatch, tmp_path, make_pdf):
    pipeline = _install_test_pipeline(monkeypatch, tmp_path)
    pipeline.ingest_file(make_pdf(["alpha beta"]))
    out = tmp_path / "corpus.jsonl"
    result = runner.invoke(cli_module.app, ["export", str(out)])
    assert result.exit_code == 0
    assert out.is_file()
    assert out.read_text(encoding="utf-8").strip()
