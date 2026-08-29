"""Crash-safe checkpointing: completed profiles are appended to disk immediately."""

from __future__ import annotations

import json
import os
import threading
from typing import Optional


class CheckpointStore:
    """One JSONL line per completed profile; reloadable to skip done work on resume."""

    def __init__(self, run_dir: str, filename: str = "partials.jsonl"):
        self.path = os.path.join(run_dir, filename)
        self._lock = threading.Lock()
        self._records: dict[int, dict] = {}
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        self._records[int(rec["profile_id"])] = rec
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue  # tolerate a torn last line after a crash

    def has(self, profile_id: int) -> bool:
        return int(profile_id) in self._records

    def append(self, record: dict) -> None:
        pid = int(record["profile_id"])
        with self._lock:
            self._records[pid] = record
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=float) + "\n")

    @property
    def records(self) -> list[dict]:
        return [self._records[pid] for pid in sorted(self._records)]

    def __len__(self) -> int:
        return len(self._records)


class ResponseAuditLog:
    """Append-only JSONL audit of every LLM attempt (full raw text, not truncated)."""

    def __init__(self, run_dir: str, filename: str = "responses.jsonl"):
        self.path = os.path.join(run_dir, filename)
        self._lock = threading.Lock()

    def log(
        self,
        profile_id: int,
        attempt: int,
        raw: Optional[str],
        finish_reason: Optional[str],
        usage: dict,
        stage: str = "final",
    ) -> None:
        from datetime import datetime

        entry = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "profile_id": profile_id,
            "attempt": attempt,
            "stage": stage,
            "finish_reason": finish_reason,
            "usage": usage,
            "raw": raw,
        }
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
