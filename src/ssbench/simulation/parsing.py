"""Response parsing and validation, ported from SSDataBench's generation scripts."""

from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np
import pandas as pd

from ssbench.datasets.schema import DatasetSpec


def stringify_allowed(spec: Any) -> str:
    if isinstance(spec, list):
        return "[" + ", ".join([f'"{v}"' for v in spec]) + "]"
    if isinstance(spec, dict) and spec.get("type") == "numeric":
        rng = f"[{spec.get('min')}, {spec.get('max')}]" if spec.get("min") is not None else "any"
        special = spec.get("special", [])
        if special:
            return f"number in {rng} OR one of {special}"
        return f"number in {rng}"
    return str(spec)


def load_json_safely(raw_text: Optional[str]) -> dict:
    if raw_text is None:
        raise json.JSONDecodeError("Empty content", "", 0)
    text = raw_text.strip()
    if not text:
        raise json.JSONDecodeError("Empty content", raw_text, 0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end >= start:
            return json.loads(text[start : end + 1])
        raise


def validate_cat(value, allowed_list) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if s in allowed_list:
        return s
    lowered = {str(a).lower(): a for a in allowed_list}
    return lowered.get(s.lower(), None)


def validate_numeric(value, spec: dict):
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        lowmap = {str(x).lower(): x for x in spec.get("special", [])}
        if s.lower() in lowmap:
            return lowmap[s.lower()]
    try:
        v = float(value)
    except Exception:
        return None
    v = max(v, spec.get("min", v))
    v = min(v, spec.get("max", v))
    return int(round(v))


def _validate_value(value, spec: Any):
    if isinstance(spec, list):
        return validate_cat(value, spec)
    if isinstance(spec, dict) and spec.get("type") == "numeric":
        return validate_numeric(value, spec)
    return value


def is_empty_value(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return False


def record_has_empty(record: dict, input_vars: list[str]) -> bool:
    for key, value in record.items():
        if key == "profile_id" or key in input_vars:
            continue
        if is_empty_value(value):
            return True
    return False


def build_record(js: dict, sampled_inputs: dict, spec: DatasetSpec) -> dict:
    """Merge LLM JSON output with conditioning inputs into one flat record."""
    record = dict(sampled_inputs)
    record["profile_id"] = sampled_inputs.get("profile_id")

    for k in spec.static_outputs:
        record[k] = _validate_value(js.get(k, None), spec.allowed(k))

    life_traj = js.get("life_trajectory", {})
    if isinstance(life_traj, dict):
        for age_str, subobj in life_traj.items():
            try:
                age = int(age_str)
            except (TypeError, ValueError):
                continue
            if not isinstance(subobj, dict):
                continue
            for k in spec.sequential_outputs:
                if k not in subobj:
                    continue
                record[f"{k}_{age}"] = _validate_value(subobj[k], spec.allowed(k))

    return record


def ordered_output_columns(spec: DatasetSpec, records: list[dict]) -> list[str]:
    """profile_id + inputs + outputs (+ any flattened sequential + derived)."""
    seen: list[str] = []
    wanted = (
        ["profile_id"]
        + spec.input_names
        + spec.static_outputs
        + spec.postprocess_modules
    )
    extra = sorted(
        {k for r in records for k in r} - set(wanted)
    )
    for col in wanted + extra:
        if col not in seen:
            seen.append(col)
    return seen


def records_to_frame(records: list[dict], spec: DatasetSpec) -> pd.DataFrame:
    df = pd.DataFrame(records)
    for col in ordered_output_columns(spec, records):
        if col not in df.columns:
            df[col] = np.nan
    df = df[ordered_output_columns(spec, records)]
    df = df.sort_values("profile_id").reset_index(drop=True)
    return df
