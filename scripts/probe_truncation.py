#!/usr/bin/env python3
"""Probe: can the model emit one complete CFPS life_trajectory JSON without truncation?

Builds the real CFPS prompt for one sampled individual, calls the model with a
generous max_tokens budget, then checks: finish_reason, usage, JSON validity,
age coverage (14..45), and per-age key completeness.
"""

from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, "src")

from ssbench.datasets.registry import load_dataset
from ssbench.llm.client import SYSTEM_JSON, LLMClient
from ssbench.preprocessing.prepare import processed_sample_path
from ssbench.settings import get_settings
from ssbench.simulation.parsing import build_record, load_json_safely, record_has_empty
from ssbench.simulation.prompts import build_prompt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="cfps")
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--profile", type=int, default=0)
    ap.add_argument("--dump", default=None, help="write raw response to this file")
    args = ap.parse_args()

    settings = get_settings()
    spec = load_dataset(args.dataset)

    import pandas as pd

    inputs = pd.read_csv(processed_sample_path(args.dataset))
    row = inputs.iloc[args.profile]
    sampled = {k: row[k] for k in spec.input_names}
    sampled["profile_id"] = int(row["profile_id"])
    prompt = build_prompt(spec, sampled)
    print(f"prompt chars: {len(prompt)}")

    client = LLMClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=args.model or settings.llm_model,
        max_tokens=args.max_tokens,
    )
    resp = client.chat(SYSTEM_JSON, prompt)
    print(f"finish_reason: {resp.finish_reason}")
    print(f"usage: {resp.usage}")
    print(f"attempts: {resp.attempts}")
    if resp.content is None:
        print("NO CONTENT")
        return 1
    print(f"content chars: {len(resp.content)}")
    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            f.write(resp.content)
        print(f"raw dumped -> {args.dump}")

    try:
        js = load_json_safely(resp.content)
    except Exception as e:  # noqa: BLE001
        print(f"JSON PARSE FAILED: {e}")
        return 1
    print("json parse: OK")

    traj = js.get("life_trajectory", {})
    ages = sorted(int(a) for a in traj if str(a).isdigit())
    lo, hi = spec.age_range
    expected = set(range(lo, hi + 1))
    missing = expected - set(ages)
    print(f"life_trajectory ages present: {len(ages)} ({ages[0] if ages else '-'}..{ages[-1] if ages else '-'})")
    print(f"missing ages: {sorted(missing) if missing else 'none'}")

    seq_vars = spec.sequential_outputs
    incomplete_ages = [
        a for a in ages
        if any(v not in traj[str(a)] for v in seq_vars)
    ]
    print(f"ages missing sequential keys: {incomplete_ages[:10] if incomplete_ages else 'none'}")

    record = build_record(js, sampled, spec)
    empty = record_has_empty(record, spec.input_names)
    print(f"record complete (no empty outputs): {not empty}")
    missing_static = [k for k in spec.static_outputs if record.get(k) is None]
    print(f"missing static outputs: {missing_static or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
