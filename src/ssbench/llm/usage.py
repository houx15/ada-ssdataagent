"""Shared token/cost ledger and hard budget guard for paid experiment runs."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


class BudgetExceededError(RuntimeError):
    """Raised before a new request when the configured USD budget is exhausted."""


def _number(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def usage_cost(usage: dict[str, Any]) -> float:
    """Return provider-reported cost, or a token-price estimate from env."""
    for key in ("cost", "total_cost", "cost_usd"):
        if key in usage:
            return _number(usage[key])
    prompt = _number(usage.get("prompt_tokens") or usage.get("input_tokens"))
    completion = _number(
        usage.get("completion_tokens") or usage.get("output_tokens")
    )
    input_per_m = _number(os.getenv("SSBENCH_INPUT_USD_PER_M"))
    output_per_m = _number(os.getenv("SSBENCH_OUTPUT_USD_PER_M"))
    return (prompt * input_per_m + completion * output_per_m) / 1_000_000


def ledger_total(path: str | os.PathLike[str]) -> float:
    total = 0.0
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    total += _number(json.loads(line).get("cost_usd"))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return total


def assert_budget_available() -> None:
    ledger = os.getenv("SSBENCH_COST_LEDGER")
    budget = _number(os.getenv("SSBENCH_BUDGET_USD"))
    if not ledger or budget <= 0:
        return
    with _LOCK:
        spent = ledger_total(ledger)
    if spent >= budget:
        raise BudgetExceededError(
            f"experiment budget exhausted: spent ${spent:.4f} of ${budget:.2f}"
        )


def record_usage(
    *,
    requested_model: str,
    resolved_model: str | None,
    response_id: str | None,
    usage: dict[str, Any],
) -> None:
    """Append one response's usage to the configured ledger, if enabled."""
    ledger = os.getenv("SSBENCH_COST_LEDGER")
    if not ledger:
        return
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "requested_model": requested_model,
        "resolved_model": resolved_model,
        "response_id": response_id,
        "prompt_tokens": int(_number(
            usage.get("prompt_tokens") or usage.get("input_tokens")
        )),
        "completion_tokens": int(_number(
            usage.get("completion_tokens") or usage.get("output_tokens")
        )),
        "cost_usd": usage_cost(usage),
    }
    path = Path(ledger)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def budgeted_chat_guard() -> None:
    """Check the shared ledger immediately before a provider request."""
    assert_budget_available()
