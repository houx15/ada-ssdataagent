"""Thin wrapper around an OpenAI-compatible chat endpoint (vLLM etc.)."""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import OpenAI

from ssbench.llm.usage import BudgetExceededError, budgeted_chat_guard, record_usage

SYSTEM_JSON = "You are a strict JSON generator. Output valid JSON only."

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


@dataclass
class LLMResponse:
    content: Optional[str]
    finish_reason: Optional[str]
    usage: dict[str, Any] = field(default_factory=dict)
    attempts: int = 1
    requested_model: Optional[str] = None
    resolved_model: Optional[str] = None
    response_id: Optional[str] = None
    error: Optional[str] = None

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
        last_attempt = 0
        for attempt in range(1, self.max_retries + 1):
            last_attempt = attempt
            try:
                budgeted_chat_guard()
                extra_body = {}
                raw_extra = os.getenv("SSBENCH_LLM_EXTRA_BODY")
                if raw_extra:
                    parsed = json.loads(raw_extra)
                    if not isinstance(parsed, dict):
                        raise ValueError("SSBENCH_LLM_EXTRA_BODY must be a JSON object")
                    extra_body = parsed
                request_kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                }
                if not _env_flag("SSBENCH_LLM_OMIT_SAMPLING_PARAMS"):
                    request_kwargs.update(
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )
                if self.json_mode:
                    request_kwargs["response_format"] = {"type": "json_object"}
                if extra_body:
                    request_kwargs["extra_body"] = extra_body
                resp = self._client.chat.completions.create(
                    **request_kwargs,
                )
                choice = resp.choices[0]
                usage: dict[str, Any] = {}
                if getattr(resp, "usage", None) is not None:
                    if hasattr(resp.usage, "model_dump"):
                        usage = resp.usage.model_dump(exclude_none=True)
                    else:
                        usage = {
                            "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
                            "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
                        }
                resolved_model = getattr(resp, "model", None)
                response_id = getattr(resp, "id", None)
                record_usage(
                    requested_model=self.model,
                    resolved_model=resolved_model,
                    response_id=response_id,
                    usage=usage,
                )
                content = _extract_content(choice.message)
                return LLMResponse(
                    content=content,
                    finish_reason=choice.finish_reason,
                    usage=usage,
                    attempts=attempt,
                    requested_model=self.model,
                    resolved_model=resolved_model,
                    response_id=response_id,
                )
            except Exception as e:  # noqa: BLE001 - deliberate broad catch for retry
                if isinstance(e, BudgetExceededError):
                    raise
                if not _is_retryable(e) or attempt == self.max_retries:
                    if isinstance(e, Exception):
                        last_error = e
                        break
                time.sleep(self.retry_delay * (2 ** (attempt - 1)) * (0.5 + random.random()))
        error = None
        if last_error is not None:
            error = f"{type(last_error).__name__}: {last_error}"
        return LLMResponse(
            content=None,
            finish_reason="error",
            attempts=last_attempt,
            error=error,
        )


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


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
