"""Minimal Ollama chat client.

Plain HTTP on purpose. The only endpoint used is POST /api/chat, and adding an
SDK for one request shape would be more dependency than value.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

# Statuses a reverse proxy returns when the origin was too slow or briefly
# unavailable — never the model rejecting the request. Retrying is safe.
RETRYABLE_STATUSES = frozenset({502, 503, 504, 520, 521, 522, 523, 524})


class LLMError(Exception):
    """Generation failed. Callers surface this rather than crashing."""


@dataclass(frozen=True)
class LLMResponse:
    content: str
    thinking: str  # gemma4 returns reasoning separately; never shown to users
    prompt_tokens: int
    output_tokens: int
    duration_seconds: float

    @property
    def tokens_per_second(self) -> float:
        if self.duration_seconds <= 0:
            return 0.0
        return self.output_tokens / self.duration_seconds


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: int = 300,
        temperature: float = 0.2,
        max_retries: int = 1,
        retry_delay_seconds: float = 3.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

    def list_models(self) -> list[str]:
        return [m["name"] for m in self._post_get("/api/tags").get("models", [])]

    def _post_get(self, path: str) -> dict:
        """GET helper — same headers, same error translation."""
        request = urllib.request.Request(
            f"{self.base_url}{path}", headers={"User-Agent": "ragforge/1.0"}
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            raise LLMError(f"could not list models at {self.base_url}: {exc}") from exc

    def _stream_with_retry(self, path: str, payload: dict) -> dict:
        """Stream, retrying when the proxy times out waiting for the origin.

        Ollama unloads an idle model, and reloading a multi-gigabyte one takes
        longer than a reverse proxy will wait — the request dies with a 524
        while the load carries on server-side. So the first call after a quiet
        period fails and the next succeeds. Retrying turns that into a slow
        answer instead of an error the reader has to interpret.

        Only one retry: it recovers a cold model, which is a real and common
        case. More than that just multiplies the wait before the reader sees an
        error that retrying was never going to fix — such as a proxy whose
        timeout is shorter than the model needs to read a long prompt.
        """
        last: LLMError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._stream(path, payload)
            except LLMError as exc:
                last = exc
                if not getattr(exc, "retryable", False):
                    raise
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_seconds)
        raise last

    def _stream(self, path: str, payload: dict) -> dict:
        """Consume a streamed Ollama reply, returning a single response dict.

        Streaming is not for progressive display — it is what keeps the request
        alive. A reverse proxy in front of the model (Cloudflare here) closes an
        idle connection after ~100s with a 524, and a 12B model answering from
        several passages routinely takes longer than that to finish. Streaming
        makes the first byte arrive in seconds, so the connection is never idle.
        """
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "ragforge/1.0",
            },
        )

        content_parts: list[str] = []
        thinking_parts: list[str] = []
        final: dict = {}
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as resp:
                for line in resp:
                    line = line.strip()
                    if not line:
                        continue
                    chunk = json.loads(line.decode("utf-8"))
                    if chunk.get("error"):
                        raise LLMError(f"{self.model}: {chunk['error']}")
                    message = chunk.get("message") or {}
                    content_parts.append(message.get("content") or "")
                    thinking_parts.append(message.get("thinking") or "")
                    if chunk.get("done"):
                        final = chunk
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:200].strip()
            if exc.code in RETRYABLE_STATUSES:
                error = LLMError(
                    f"{self.base_url} returned {exc.code} — the model was probably "
                    f"still loading. Retrying."
                )
                error.retryable = True
                raise error from exc
            raise LLMError(f"{self.base_url} returned {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"could not reach {self.base_url}: {exc.reason}") from exc
        except TimeoutError as exc:
            error = LLMError(
                f"{self.model} did not respond within {self.timeout_seconds}s"
            )
            error.retryable = True
            raise error from exc
        except json.JSONDecodeError as exc:
            raise LLMError(f"malformed stream from {self.base_url}: {exc}") from exc

        if not final:
            # No chunk carried done=true, so the stream was cut short rather
            # than finishing. Blaming the model for "an empty response" here
            # sends you looking at prompts when the endpoint dropped.
            error = LLMError(
                f"the stream from {self.base_url} ended before the model "
                f"finished — the connection dropped or the model was still loading"
            )
            error.retryable = True
            raise error

        final["message"] = {
            "role": "assistant",
            "content": "".join(content_parts),
            "thinking": "".join(thinking_parts),
        }
        return final

    def chat_json(
        self, system: str, user: str, schema: dict, num_ctx: int = 8192
    ) -> tuple[dict, LLMResponse]:
        """One structured request. Returns (parsed object, response metadata).

        `schema` is passed as Ollama's `format`, which constrains decoding — the
        reply is valid JSON matching the schema rather than prose to be salvaged.
        """
        raw = self._stream_with_retry(
            "/api/chat",
            {
                "model": self.model,
                "stream": True,
                "format": schema,
                # num_ctx is deliberately modest. Asking for 16384 alongside an
                # 8.4GB model made the server stall long enough that the proxy
                # gave up before a single token was emitted, while 8192 produced
                # a first byte in under a minute on the same prompt. Real
                # requests here run ~5,300 prompt tokens plus ~600 generated,
                # so a 16K window bought nothing and cost every request.
                #
                # Ask the server to hold the model in memory. Ollama unloads
                # after 5 minutes idle by default, and reloading several
                # gigabytes is what causes the first request after a pause to
                # time out. Requesting it per-call means the fix travels with
                # the client, without needing OLLAMA_KEEP_ALIVE set on the box.
                "keep_alive": "30m",
                "options": {
                    "temperature": self.temperature,
                    "num_ctx": num_ctx,
                },
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )

        message = raw.get("message") or {}
        content = message.get("content", "")
        if not content.strip():
            raise LLMError(f"{self.model} returned an empty response")

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"{self.model} returned text that is not valid JSON: {exc}"
            ) from exc

        return parsed, LLMResponse(
            content=content,
            thinking=message.get("thinking", ""),
            prompt_tokens=int(raw.get("prompt_eval_count") or 0),
            output_tokens=int(raw.get("eval_count") or 0),
            duration_seconds=float(raw.get("total_duration") or 0) / 1e9,
        )
