"""Dataset-agnostic preprocessing driven by the `preprocessing:` block in dataset yamls."""

from __future__ import annotations

import os

import pandas as pd

from ssbench.datasets.schema import DatasetSpec
from ssbench.preprocessing.derive import derive_columns
from ssbench.settings import get_settings

SAMPLE_N = 1000
SAMPLE_SEED = 42


def processed_sample_path(name: str) -> str:
    return os.path.join(get_settings().processed_dir, name, "sample.csv")


def prepare_dataset(spec: DatasetSpec, n: int = SAMPLE_N, seed: int = SAMPLE_SEED) -> str:
    """Clean + derive + sample the raw real data; write ``data/processed/<name>/sample.csv``.

    Order matters: derivations consume wide yearly columns, so they must run
    before the final column selection drops them.
    """
    cfg = spec.preprocessing
    source = os.path.join(get_settings().repo_root, cfg["source"])
    df = pd.read_csv(source, low_memory=False)

    if cfg.get("drop_first_column"):
        first = df.columns[0]
        if first.startswith("Unnamed") or first == "":
            df = df.drop(columns=[first])

    if cfg.get("require_complete_inputs"):
        before = len(df)
        df = df.dropna(subset=[c for c in spec.input_names if c in df.columns])
        print(f"[prepare:{spec.name}] complete-input filter: {before} -> {len(df)} rows")

    if cfg.get("derived_columns"):
        df = derive_columns(df, cfg["derived_columns"])

    wanted = spec.input_names + list(cfg.get("static_columns", spec.static_outputs))
    wanted += list(cfg.get("derived_columns", {}).keys())
    missing = [c for c in spec.input_names if c not in df.columns]
    if missing:
        raise ValueError(f"[prepare:{spec.name}] input columns missing from raw data: {missing}")
    absent = set(wanted) - set(df.columns)
    if absent:
        print(f"[prepare:{spec.name}] note: wanted columns absent from raw data: {sorted(absent)}")
    df = df[[c for c in wanted if c in df.columns]].copy()

    n_eff = min(n, len(df))
    df = df.sample(n=n_eff, replace=False, random_state=seed).reset_index(drop=True)
    df.insert(0, "profile_id", range(len(df)))

    out = processed_sample_path(spec.name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[prepare:{spec.name}] wrote {len(df)} rows x {len(df.columns)} cols -> {out}")
    return out
