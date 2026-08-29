"""Shared series-cleaning helpers (port of SSDataBench evaluation/code_by_type/common.py)."""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd


def clean_str(value) -> Optional[str]:
    if value is None:
        return np.nan
    if isinstance(value, float) and np.isnan(value):
        return np.nan
    text = str(value).strip()
    return text if text else np.nan


def apply_value_map(series: pd.Series, value_map: Optional[Mapping]) -> pd.Series:
    if not value_map:
        return series
    lower_map = {str(k).strip().lower(): v for k, v in value_map.items()}

    def _map_one(v):
        if pd.isna(v):
            return v
        return lower_map.get(str(v).strip().lower(), v)

    return series.map(_map_one)


def drop_values(series: pd.Series, drops: Optional[Iterable]) -> pd.Series:
    if not drops:
        return series
    drop_set = {str(d).strip() for d in drops if d not in (None, "", np.nan)}
    return series.mask(series.astype(str).str.strip().isin(drop_set), other=np.nan)


def to_numeric_clean(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def prep_variable(df: pd.DataFrame, var: str, cfg: dict) -> pd.Series:
    """Port of type2/_prep: clean -> map -> drop -> numeric -> log1p."""
    s = df[var].map(clean_str)
    s = apply_value_map(s, cfg.get("value_map", {}))
    s = drop_values(s, cfg.get("drop_values", []))
    if (cfg.get("type") or "").lower() == "numeric":
        s = to_numeric_clean(s)
    if cfg.get("log_transform", False) is True:
        s = pd.Series(np.where(s >= 0, np.log1p(s), np.nan), index=s.index)
    return s


def clean_event_series(s: pd.Series, cfg: dict) -> pd.Series:
    """Port of type4/_clean_series for event-timing columns."""
    s = s.astype(str).str.strip().replace({"": np.nan, "NA": np.nan, "nan": np.nan})
    if "value_map" in cfg:
        for k, v in cfg["value_map"].items():
            s = s.replace(k, v)
    s = s.replace(["inf", "Inf", "INF"], np.inf)
    return pd.to_numeric(s, errors="coerce")


def clean_regression_series(s: pd.Series, cfg: dict) -> pd.Series:
    """Port of type3/_clean_series for regression columns."""
    s = s.copy()
    s = s.astype(str).str.strip().replace({"": np.nan, "NA": np.nan, "nan": np.nan})
    if "value_map" in cfg and isinstance(cfg["value_map"], dict):
        lower_map = {str(k).lower(): v for k, v in cfg["value_map"].items()}
        s = s.map(lambda x: lower_map.get(x.lower(), x) if isinstance(x, str) else x)
    if "drop_values" in cfg and cfg["drop_values"]:
        drops = set(str(d).strip() for d in cfg["drop_values"])
        s = s.mask(s.astype(str).isin(drops))
    if (cfg.get("type") or "").lower() == "numeric":
        s = pd.to_numeric(s, errors="coerce")
    elif (cfg.get("type") or "").lower() == "categorical":
        s = s.astype("category")
    if cfg.get("log_transform", False) is True:
        s = np.where(s >= 0, np.log1p(s), np.nan)
    return s
