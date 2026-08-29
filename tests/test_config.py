import pytest

from ragforge.config import Settings


def test_defaults_match_spec():
    s = Settings()
    assert s.embedding_model_name == "BAAI/bge-small-en-v1.5"
    assert s.chunk_size == 400
    assert s.chunk_overlap == 60
    assert s.max_model_tokens == 510


def test_chunk_size_must_fit_the_model_window():
    with pytest.raises(ValueError):
        Settings(chunk_size=600)


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        Settings(chunk_size=100, chunk_overlap=100)


def test_derived_directories_hang_off_data_dir(tmp_path):
    s = Settings(data_dir=tmp_path)
    assert s.uploads_dir == tmp_path / "uploads"
    assert s.chroma_dir == tmp_path / "chroma"


def test_ensure_dirs_creates_them(tmp_path):
    s = Settings(data_dir=tmp_path / "d")
    s.ensure_dirs()
    assert s.uploads_dir.is_dir()
    assert s.chroma_dir.is_dir()


def test_env_override(monkeypatch):
    monkeypatch.setenv("RAGFORGE_CHUNK_SIZE", "250")
    assert Settings().chunk_size == 250
