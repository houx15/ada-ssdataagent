"""SSDataBench paper paradigm: one LLM call per individual (digital-twin baseline)."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd

from ssbench.datasets.schema import DatasetSpec
from ssbench.llm.client import SYSTEM_JSON, LLMClient
from ssbench.simulation.checkpoint import CheckpointStore, ResponseAuditLog
from ssbench.simulation.methods.base import register_method
from ssbench.simulation.parsing import (
    build_record,
    load_json_safely,
    record_has_empty,
    records_to_frame,
)
from ssbench.simulation.lifecycle import apply_postprocess
from ssbench.simulation.prompts import build_prompt


@register_method
class DirectGeneration:
    name = "direct"

    def __init__(self, client: LLMClient, max_attempts: int = 4, retry_delay: float = 1.0):
        self.client = client
        self.max_attempts = max_attempts
        self.retry_delay = retry_delay
        self._lock = threading.Lock()
        self._done = 0

    def generate(
        self,
        spec: DatasetSpec,
        inputs_df: pd.DataFrame,
        failure_logger=None,
        checkpoint: Optional[CheckpointStore] = None,
    ) -> pd.DataFrame:
        total = len(inputs_df)
        if checkpoint is not None and len(checkpoint):
            print(f"[direct] resume: {len(checkpoint)} profiles already checkpointed, skipping them")
            inputs_df = inputs_df[~inputs_df["profile_id"].map(checkpoint.has)]

        todo = len(inputs_df)
        records: list[dict] = list(checkpoint.records) if checkpoint is not None else []
        print(f"[direct] generating {todo}/{total} profiles with {self.client.model} ...")

        start = time.time()
        with ThreadPoolExecutor(max_workers=self._concurrency(todo)) as ex:
            futures = {
                ex.submit(self._generate_one, spec, inputs_df.iloc[i], failure_logger): i
                for i in range(todo)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    record = future.result()
                except Exception as e:  # noqa: BLE001
                    print(f"[direct] profile {idx} failed entirely: {e}")
                    record = None
                if record is not None:
                    records.append(record)
                    if checkpoint is not None and _record_is_complete(record, spec):
                        checkpoint.append(record)
                with self._lock:
                    self._done += 1
                    if self._done % 10 == 0 or self._done == todo:
                        elapsed = time.time() - start
                        rate = self._done / elapsed if elapsed else 0.0
                        eta = (todo - self._done) / rate if rate else 0.0
                        print(
                            f"[direct] progress {self._done}/{todo} "
                            f"({rate:.1f}/s, eta {eta / 60:.1f}m)"
                        )

        return records_to_frame(records, spec)

    def _concurrency(self, total: int) -> int:
        import os

        from ssbench.settings import get_settings

        cap = int(os.getenv("SSBB_MAX_WORKERS", "0")) or get_settings().llm_concurrency
        return max(1, min(cap, max(total, 1)))

    def _generate_one(self, spec: DatasetSpec, row: pd.Series, failure_logger) -> dict:
        sampled_inputs = {k: row[k] for k in spec.input_names}
        sampled_inputs["profile_id"] = int(row["profile_id"])
        prompt = build_prompt(spec, sampled_inputs)

        last_record: Optional[dict] = None
        for attempt in range(1, self.max_attempts + 1):
            resp = self.client.chat(SYSTEM_JSON, prompt)
            if failure_logger is not None:
                failure_logger(
                    profile_id=sampled_inputs["profile_id"],
                    attempt=attempt,
                    raw=resp.content,
                    finish_reason=resp.finish_reason,
                    usage=resp.usage,
                    error=resp.error,
                )
            # The client has already retried transient transport/provider
            # errors.  Retrying an empty terminal error again at the profile
            # layer only multiplies configuration failures (for example, an
            # unsupported request parameter) without any chance of recovery.
            if resp.content is None and resp.finish_reason == "error":
                return build_record({}, sampled_inputs, spec)
            try:
                js = load_json_safely(resp.content)
            except Exception:  # noqa: BLE001
                last_record = None
                self._sleep()
                continue

            record = build_record(js, sampled_inputs, spec)
            if _record_is_complete(record, spec):
                return record
            last_record = record
            self._sleep()

        return last_record if last_record is not None else build_record({}, sampled_inputs, spec)

    def _sleep(self):
        if self.retry_delay > 0:
            time.sleep(self.retry_delay)


def _record_is_complete(record: dict, spec: DatasetSpec) -> bool:
    """Match checkpoint eligibility to the final run completeness check."""
    if record_has_empty(record, spec.input_names):
        return False
    frame = records_to_frame([record], spec)
    if spec.postprocess_modules:
        frame = apply_postprocess(frame, spec.postprocess_modules)
    required = [
        column
        for column in spec.static_outputs + spec.postprocess_modules
        if column in frame.columns
    ]
    return bool(required) and not bool(frame[required].isna().any(axis=1).iloc[0])
