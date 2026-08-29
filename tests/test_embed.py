import pytest

from ragforge.embed import QUERY_PREFIX, Embedder, HFTokenizer


class _StubHF:
    """Minimal stand-in for a HuggingFace tokenizer."""

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"t{i}" for i in ids)


def test_counts_tokens_without_special_tokens():
    assert HFTokenizer(_StubHF()).count_tokens("a b c") == 3


def test_split_by_tokens_bounds_every_piece():
    pieces = HFTokenizer(_StubHF()).split_by_tokens("a b c d e", 2)
    assert len(pieces) == 3
    assert all(len(p.split()) <= 2 for p in pieces)


def test_tail_tokens_returns_at_most_n_tokens():
    assert HFTokenizer(_StubHF()).tail_tokens("a b c d e", 2) == "t3 t4"


def test_tail_tokens_of_zero_is_empty():
    assert HFTokenizer(_StubHF()).tail_tokens("a b c", 0) == ""


def test_query_prefix_is_the_bge_retrieval_instruction():
    assert QUERY_PREFIX.startswith("Represent this sentence")


@pytest.mark.slow
def test_real_model_round_trip():
    """Downloads the model. Run with: pytest -m slow"""
    embedder = Embedder()
    assert embedder.dimension == 384

    vectors = embedder.embed_documents(["the cat sat", "quarterly revenue report"])
    assert len(vectors) == 2
    assert all(len(v) == 384 for v in vectors)
    assert abs(sum(c * c for c in vectors[0]) ** 0.5 - 1.0) < 1e-3

    query = embedder.embed_query("where did the cat sit")
    cat_score = sum(a * b for a, b in zip(query, vectors[0]))
    revenue_score = sum(a * b for a, b in zip(query, vectors[1]))
    assert cat_score > revenue_score


@pytest.mark.slow
def test_tokenizer_agrees_with_the_model_window():
    embedder = Embedder()
    long_text = " ".join(["word"] * 5000)
    pieces = embedder.tokenizer.split_by_tokens(long_text, 510)
    assert all(embedder.tokenizer.count_tokens(p) <= 510 for p in pieces)


@pytest.mark.slow
def test_real_tail_tokens_is_bounded():
    embedder = Embedder()
    text = " ".join(f"word{i}" for i in range(400))
    tail = embedder.tokenizer.tail_tokens(text, 60)
    assert embedder.tokenizer.count_tokens(tail) <= 60
