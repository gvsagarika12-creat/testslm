"""Minimal Ollama chat client.

Plain HTTP on purpose. The only endpoint used is POST /api/chat, and adding an
SDK for one request shape would be more dependency than value.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass


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
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature

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
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise LLMError(f"{self.base_url} returned {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"could not reach {self.base_url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise LLMError(
                f"{self.model} did not respond within {self.timeout_seconds}s"
            ) from exc
        except json.JSONDecodeError as exc:
            raise LLMError(f"malformed stream from {self.base_url}: {exc}") from exc

        if not final:
            # No chunk carried done=true, so the stream was cut short rather
            # than finishing. Blaming the model for "an empty response" here
            # sends you looking at prompts when the endpoint dropped.
            raise LLMError(
                f"the stream from {self.base_url} ended before the model "
                f"finished — the endpoint may be down or the connection dropped"
            )

        final["message"] = {
            "role": "assistant",
            "content": "".join(content_parts),
            "thinking": "".join(thinking_parts),
        }
        return final

    def chat_json(
        self, system: str, user: str, schema: dict, num_ctx: int = 16384
    ) -> tuple[dict, LLMResponse]:
        """One structured request. Returns (parsed object, response metadata).

        `schema` is passed as Ollama's `format`, which constrains decoding — the
        reply is valid JSON matching the schema rather than prose to be salvaged.
        """
        raw = self._stream(
            "/api/chat",
            {
                "model": self.model,
                "stream": True,
                "format": schema,
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
