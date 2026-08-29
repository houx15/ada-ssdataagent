"""Thin wrapper around an OpenAI-compatible chat endpoint (vLLM etc.)."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import OpenAI

SYSTEM_JSON = "You are a strict JSON generator. Output valid JSON only."

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


@dataclass
class LLMResponse:
    content: Optional[str]
    finish_reason: Optional[str]
    usage: dict[str, Any] = field(default_factory=dict)
    attempts: int = 1

    @property
    def ok(self) -> bool:
        return (
            self.content is not None
            and self.finish_reason == "stop"
        )


class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 32768,
        json_mode: bool = True,
        max_retries: int = 4,
        retry_delay: float = 2.0,
        timeout: float = 1200.0,
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.json_mode = json_mode
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def chat(self, system: str, user: str) -> LLMResponse:
        """Send one chat request, retrying transient errors with backoff."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    **({"response_format": {"type": "json_object"}} if self.json_mode else {}),
                )
                choice = resp.choices[0]
                usage = {}
                if getattr(resp, "usage", None) is not None:
                    usage = {
                        "prompt_tokens": resp.usage.prompt_tokens,
                        "completion_tokens": resp.usage.completion_tokens,
                    }
                content = _extract_content(choice.message)
                return LLMResponse(
                    content=content,
                    finish_reason=choice.finish_reason,
                    usage=usage,
                    attempts=attempt,
                )
            except Exception as e:  # noqa: BLE001 - deliberate broad catch for retry
                if not _is_retryable(e) or attempt == self.max_retries:
                    if isinstance(e, Exception):
                        last_error = e
                        break
                time.sleep(self.retry_delay * (2 ** (attempt - 1)) * (0.5 + random.random()))
        return LLMResponse(content=None, finish_reason="error", attempts=self.max_retries)


def _extract_content(message) -> Optional[str]:
    """Normalize provider-specific content shapes (str or segment list) to str."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for seg in content:
            if isinstance(seg, dict):
                for key in ("text", "content", "json", "object", "data"):
                    if seg.get(key) is not None:
                        import json as _json
                        val = seg[key]
                        parts.append(val if isinstance(val, str) else _json.dumps(val))
                        break
            else:
                parts.append(str(seg))
        return "".join(parts).strip()
    return str(content or "").strip()


def _is_retryable(e: Exception) -> bool:
    status = getattr(e, "status_code", None)
    if status is not None:
        return status in _RETRYABLE_STATUS
    name = type(e).__name__
    return name in {
        "APIConnectionError", "APITimeoutError", "InternalServerError",
        "RateLimitError", "APIStatusError",
    }
