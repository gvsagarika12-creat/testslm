import pytest

from ragforge.parse import (
    CorruptPdfError,
    EncryptedPdfError,
    NoTextLayerError,
    compute_doc_id,
    extract_pages,
    normalize_text,
)


def test_joins_words_broken_across_a_line():
    assert normalize_text("inter-\nnational") == "international"


def test_single_newlines_become_spaces():
    assert normalize_text("one\ntwo") == "one two"


def test_paragraph_breaks_survive():
    assert normalize_text("one\n\ntwo") == "one\n\ntwo"


def test_long_blank_runs_collapse_to_one_break():
    assert normalize_text("one\n\n\n\ntwo") == "one\n\ntwo"


def test_repeated_spaces_collapse():
    assert normalize_text("a     b") == "a b"


def test_extracts_one_entry_per_page_in_order(make_pdf):
    path = make_pdf(["alpha text", "beta text"])
    pages = extract_pages(path)
    assert [p.page_number for p in pages] == [1, 2]
    assert "alpha" in pages[0].text
    assert "beta" in pages[1].text


def test_blank_pages_are_dropped(make_pdf):
    path = make_pdf(["real content here", "   ", "more content"])
    pages = extract_pages(path)
    assert [p.page_number for p in pages] == [1, 3]


def test_encrypted_pdf_raises(make_pdf):
    path = make_pdf(["secret"], encrypt=True)
    with pytest.raises(EncryptedPdfError):
        extract_pages(path)


def test_pdf_with_no_text_raises(make_pdf):
    path = make_pdf(["", ""])
    with pytest.raises(NoTextLayerError):
        extract_pages(path)


def test_corrupt_file_raises(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"this is not a pdf")
    with pytest.raises(CorruptPdfError):
        extract_pages(path)


def test_doc_id_is_deterministic_and_content_sensitive():
    assert compute_doc_id(b"same") == compute_doc_id(b"same")
    assert compute_doc_id(b"same") != compute_doc_id(b"different")
    assert len(compute_doc_id(b"x")) == 64
