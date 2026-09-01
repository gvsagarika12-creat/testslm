"""Ollama client. Network is faked except for the tests marked slow."""
import io
import json
import urllib.error

import pytest

from ragforge.llm import LLMError, OllamaClient

SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}}


def fake_urlopen(monkeypatch, chunks, capture=None):
    """Patch urlopen to return an NDJSON stream, recording the request."""
    ndjson = b"".join(json.dumps(c).encode() + b"\n" for c in chunks)

    def _open(request, timeout=None):
        if capture is not None:
            capture["request"] = request
            capture["timeout"] = timeout
        return io.BytesIO(ndjson)

    monkeypatch.setattr("urllib.request.urlopen", _open)


def ok_body(content='{"a": "b"}', **extra):
    """A streamed reply: content split across chunks, then a final done chunk.

    Ollama streams newline-delimited JSON, so the client must reassemble the
    content rather than read one object.
    """
    midpoint = len(content) // 2
    final = {
        "message": {"role": "assistant", "content": content[midpoint:], "thinking": ""},
        "done": True,
        "prompt_eval_count": 120,
        "eval_count": 45,
        "total_duration": 3_000_000_000,
    }
    final.update(extra)
    return [
        {"message": {"content": content[:midpoint], "thinking": "hmm"}, "done": False},
        final,
    ]


@pytest.fixture
def client():
    return OllamaClient("https://example.test/", "gemma4:12b", timeout_seconds=42)


def test_trailing_slash_is_stripped():
    assert OllamaClient("https://x.test/", "m").base_url == "https://x.test"


def test_returns_parsed_json_and_metadata(monkeypatch, client):
    fake_urlopen(monkeypatch, ok_body())
    parsed, meta = client.chat_json("sys", "usr", SCHEMA)
    assert parsed == {"a": "b"}
    assert meta.prompt_tokens == 120
    assert meta.output_tokens == 45
    assert meta.duration_seconds == 3.0


def test_tokens_per_second(monkeypatch, client):
    fake_urlopen(monkeypatch, ok_body())
    _, meta = client.chat_json("sys", "usr", SCHEMA)
    assert meta.tokens_per_second == 15.0


def test_thinking_is_captured_separately(monkeypatch, client):
    """gemma4 returns reasoning apart from content; it must not leak into it."""
    fake_urlopen(monkeypatch, ok_body())
    _, meta = client.chat_json("sys", "usr", SCHEMA)
    assert meta.thinking == "hmm"
    assert "hmm" not in meta.content


def test_content_split_across_chunks_is_reassembled(monkeypatch, client):
    """The whole point of streaming: the JSON arrives in pieces."""
    fake_urlopen(monkeypatch, [
        {"message": {"content": '{"a":'}, "done": False},
        {"message": {"content": ' "spl'}, "done": False},
        {"message": {"content": 'it"}'}, "done": True, "eval_count": 3},
    ])
    parsed, _ = client.chat_json("sys", "usr", SCHEMA)
    assert parsed == {"a": "split"}


def test_an_error_chunk_mid_stream_is_raised(monkeypatch, client):
    fake_urlopen(monkeypatch, [
        {"message": {"content": "{"}, "done": False},
        {"error": "model not found"},
    ])
    with pytest.raises(LLMError, match="model not found"):
        client.chat_json("sys", "usr", SCHEMA)


def test_sends_a_user_agent(monkeypatch, client):
    """Cloudflare 403s the default Python agent, so this is load-bearing."""
    capture = {}
    fake_urlopen(monkeypatch, ok_body(), capture)
    client.chat_json("sys", "usr", SCHEMA)
    assert capture["request"].get_header("User-agent") == "ragforge/1.0"


def test_sends_model_schema_and_messages(monkeypatch, client):
    capture = {}
    fake_urlopen(monkeypatch, ok_body(), capture)
    client.chat_json("SYS", "USR", SCHEMA)
    sent = json.loads(capture["request"].data)
    assert sent["model"] == "gemma4:12b"
    assert sent["format"] == SCHEMA
    assert sent["stream"] is True, "streaming keeps the proxy from timing out"
    assert sent["messages"][0] == {"role": "system", "content": "SYS"}
    assert sent["messages"][1] == {"role": "user", "content": "USR"}


def test_uses_the_configured_timeout(monkeypatch, client):
    capture = {}
    fake_urlopen(monkeypatch, ok_body(), capture)
    client.chat_json("sys", "usr", SCHEMA)
    assert capture["timeout"] == 42


def test_http_error_is_wrapped_with_the_status(monkeypatch, client):
    def _raise(request, timeout=None):
        raise urllib.error.HTTPError(
            "u", 403, "Forbidden", {}, io.BytesIO(b"blocked by cloudflare")
        )

    monkeypatch.setattr("urllib.request.urlopen", _raise)
    with pytest.raises(LLMError, match="403"):
        client.chat_json("sys", "usr", SCHEMA)


def test_unreachable_host_is_wrapped(monkeypatch, client):
    def _raise(request, timeout=None):
        raise urllib.error.URLError("name resolution failed")

    monkeypatch.setattr("urllib.request.urlopen", _raise)
    with pytest.raises(LLMError, match="could not reach"):
        client.chat_json("sys", "usr", SCHEMA)


def test_empty_response_is_an_error(monkeypatch, client):
    fake_urlopen(monkeypatch, ok_body(content="   "))
    with pytest.raises(LLMError, match="empty response"):
        client.chat_json("sys", "usr", SCHEMA)


def test_non_json_content_is_an_error(monkeypatch, client):
    fake_urlopen(monkeypatch, ok_body(content="I'm sorry, I cannot"))
    with pytest.raises(LLMError, match="not valid JSON"):
        client.chat_json("sys", "usr", SCHEMA)


def test_missing_counters_default_to_zero(monkeypatch, client):
    fake_urlopen(monkeypatch, [{"message": {"content": '{"a":"b"}'}, "done": True}])
    _, meta = client.chat_json("sys", "usr", SCHEMA)
    assert meta.output_tokens == 0
    assert meta.tokens_per_second == 0.0


# --- the real endpoint ------------------------------------------------------

@pytest.mark.slow
def test_real_endpoint_lists_the_configured_model():
    from ragforge.config import settings

    live = OllamaClient(settings.ollama_base_url, settings.ollama_model)
    assert settings.ollama_model in live.list_models()


@pytest.mark.slow
def test_real_endpoint_honours_a_json_schema():
    from ragforge.config import settings

    live = OllamaClient(settings.ollama_base_url, settings.ollama_model)
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    parsed, meta = live.chat_json(
        system="Reply with the single word OK in the answer field.",
        user="Say OK.",
        schema=schema,
    )
    assert "answer" in parsed
    assert meta.output_tokens > 0


def test_a_truncated_stream_says_so_rather_than_blaming_the_model(monkeypatch, client):
    """No done chunk means the connection dropped, not that the model was silent."""
    fake_urlopen(monkeypatch, [
        {"message": {"content": '{"a": "par'}, "done": False},
    ])
    with pytest.raises(LLMError, match="ended before the model finished"):
        client.chat_json("sys", "usr", SCHEMA)


def test_a_completed_stream_with_no_content_still_reports_an_empty_response(monkeypatch, client):
    """A finished stream that produced nothing is a genuine model failure."""
    fake_urlopen(monkeypatch, [{"message": {"content": ""}, "done": True}])
    with pytest.raises(LLMError, match="empty response"):
        client.chat_json("sys", "usr", SCHEMA)
