"""Derivation of life-course summary columns from wide yearly columns.

Semantics are identical to SSDataBench's simulation/postprocess_utils.py so that
real and simulated data are aggregated the same way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mode_over_ages(df: pd.DataFrame, prefix: str, ages: tuple[int, int]) -> pd.Series:
    """Most frequent value across ``<prefix>_<age>`` columns for ages in [lo, hi]."""
    cols = [c for c in df.columns if c.startswith(prefix + "_")]
    cols = [
        c for c in cols
        if ages[0] <= int(c.rsplit("_", 1)[1]) <= ages[1]
    ]
    if not cols:
        return pd.Series(np.nan, index=df.index)

    def _mode(row):
        vals = [v for v in row[cols] if pd.notna(v)]
        if not vals:
            return np.nan
        return pd.Series(vals).mode().iloc[0]

    return df.apply(_mode, axis=1)


def mean_over_ages(df: pd.DataFrame, prefix: str, ages: tuple[int, int]) -> pd.Series:
    """Mean over ``<prefix>_<age>`` columns for ages in [lo, hi], ignoring NaN."""
    cols = [c for c in df.columns if c.startswith(prefix + "_")]
    cols = [
        c for c in cols
        if ages[0] <= int(c.rsplit("_", 1)[1]) <= ages[1]
    ]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    return df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)


DERIVERS = {
    "mode_over_ages": mode_over_ages,
    "mean_over_ages": mean_over_ages,
}


def derive_columns(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    for name, rule in spec.items():
        func = DERIVERS.get(rule["kind"])
        if func is None:
            raise ValueError(f"Unknown derivation kind: {rule['kind']}")
        df[name] = func(df, rule["prefix"], tuple(rule["ages"]))
    return df
