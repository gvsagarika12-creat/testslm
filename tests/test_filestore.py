import pytest

from ragforge.config import Settings
from ragforge.filestore import FileStoreError, LocalFileStore, build_file_store


@pytest.fixture
def store(tmp_path):
    return LocalFileStore(tmp_path / "uploads")


def test_creates_its_directory(tmp_path):
    root = tmp_path / "nested" / "uploads"
    LocalFileStore(root)
    assert root.is_dir()


def test_save_writes_the_bytes(store, tmp_path):
    store.save("doc.pdf", b"hello")
    assert (tmp_path / "uploads" / "doc.pdf").read_bytes() == b"hello"


def test_save_returns_a_usable_uri(store):
    uri = store.save("doc.pdf", b"hello")
    assert uri.startswith("file:///")
    assert uri.endswith("doc.pdf")


def test_saving_the_same_name_replaces_it(store):
    store.save("doc.pdf", b"first")
    store.save("doc.pdf", b"second")
    assert store.read("doc.pdf") == b"second"
    assert store.list_files() == ["doc.pdf"]


def test_exists_and_read_round_trip(store):
    assert not store.exists("doc.pdf")
    store.save("doc.pdf", b"content")
    assert store.exists("doc.pdf")
    assert store.read("doc.pdf") == b"content"


def test_list_files_is_sorted(store):
    for name in ("c.pdf", "a.pdf", "b.pdf"):
        store.save(name, b"x")
    assert store.list_files() == ["a.pdf", "b.pdf", "c.pdf"]


def test_no_partial_files_are_left_behind(store, tmp_path):
    store.save("doc.pdf", b"x" * 1000)
    leftovers = [p.name for p in (tmp_path / "uploads").iterdir()]
    assert leftovers == ["doc.pdf"], f"temp files left behind: {leftovers}"


def test_directory_traversal_is_stripped(store, tmp_path):
    # A crafted name must not escape the store root.
    store.save("../../escaped.pdf", b"x")
    assert not (tmp_path / "escaped.pdf").exists()
    assert store.list_files() == ["escaped.pdf"]


def test_empty_filename_is_rejected(store):
    with pytest.raises(FileStoreError):
        store.save("", b"x")


def test_location_describes_the_directory(store, tmp_path):
    assert store.location == str(tmp_path / "uploads")


def test_build_file_store_returns_local(tmp_path):
    config = Settings(data_dir=tmp_path / "data")
    assert isinstance(build_file_store(config), LocalFileStore)


def test_build_file_store_uses_the_uploads_dir(tmp_path):
    config = Settings(data_dir=tmp_path / "data")
    assert build_file_store(config).location == str(config.uploads_dir)


def test_unknown_backend_is_rejected(tmp_path):
    config = Settings(data_dir=tmp_path / "data")
    # Bypass validation to simulate a bad value reaching the factory.
    object.__setattr__(config, "file_store_backend", "ftp")
    with pytest.raises(ValueError, match="unknown file store backend"):
        build_file_store(config)
