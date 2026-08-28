import json

from ragforge.export import export_jsonl
from ragforge.store import ChromaStore
from tests.test_store import make_chunk, unit


def test_exports_one_json_object_per_chunk(tmp_path):
    store = ChromaStore(tmp_path / "chroma", "test")
    store.upsert(
        [make_chunk(0, text="first chunk"), make_chunk(1, text="second chunk")],
        [unit(1, 0), unit(0, 1)],
    )
    out = tmp_path / "corpus.jsonl"
    written = export_jsonl(store, out)

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert written == 2
    assert len(lines) == 2

    record = json.loads(lines[0])
    assert record == {
        "text": "first chunk",
        "source": "a.pdf",
        "page_start": 1,
        "page_end": 1,
    }


def test_exporting_an_empty_store_writes_an_empty_file(tmp_path):
    store = ChromaStore(tmp_path / "chroma", "test")
    out = tmp_path / "corpus.jsonl"
    assert export_jsonl(store, out) == 0
    assert out.read_text(encoding="utf-8") == ""


def test_export_handles_non_ascii(tmp_path):
    store = ChromaStore(tmp_path / "chroma", "test")
    store.upsert([make_chunk(0, text="café — naïve")], [unit(1, 0)])
    out = tmp_path / "corpus.jsonl"
    export_jsonl(store, out)
    assert json.loads(out.read_text(encoding="utf-8"))["text"] == "café — naïve"
