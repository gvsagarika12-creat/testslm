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

    def _post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                # Cloudflare fronts this endpoint and rejects the default
                # "Python-urllib/x.y" agent with a 403 that looks like an auth
                # failure. An explicit agent is required, not cosmetic.
                "User-Agent": "ragforge/1.0",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as resp:
                return json.loads(resp.read().decode("utf-8"))
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
            raise LLMError(f"malformed response from {self.base_url}: {exc}") from exc

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

    def chat_json(
        self, system: str, user: str, schema: dict, num_ctx: int = 16384
    ) -> tuple[dict, LLMResponse]:
        """One structured request. Returns (parsed object, response metadata).

        `schema` is passed as Ollama's `format`, which constrains decoding — the
        reply is valid JSON matching the schema rather than prose to be salvaged.
        """
        raw = self._post(
            "/api/chat",
            {
                "model": self.model,
                "stream": False,
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
