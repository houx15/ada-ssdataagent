"""T1 — univariate distribution comparison (chi-square for categorical, KS for numeric)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, entropy, ks_2samp

from ssbench.evaluation.bootstrap import BootstrapConfig, RngPair
from ssbench.evaluation.cleaning import prep_variable


def _bootstrap_cat(r: pd.Series, s: pd.Series, boot: BootstrapConfig, rng: RngPair) -> float:
    r = r.dropna()
    s = s.dropna()
    if r.empty or s.empty:
        return np.nan
    cats = sorted(set(r) | set(s))
    p_r = r.value_counts().reindex(cats, fill_value=0).to_numpy() / len(r)
    p_s = s.value_counts().reindex(cats, fill_value=0).to_numpy() / len(s)
    not_sig = 0
    for _ in range(boot.B):
        rb = rng._rng.multinomial(boot.sample_n, p_r)
        sb = rng._rng.multinomial(boot.sample_n, p_s)
        obs = np.vstack([rb, sb])
        obs = obs[:, obs.sum(axis=0) > 0]  # union of sampled categories, as in the original
        try:
            _, p, _, _ = chi2_contingency(obs, correction=False)
        except ValueError:
            continue
        if p > boot.alpha:
            not_sig += 1
    return not_sig / boot.B if boot.B else np.nan


def _bootstrap_num(r: pd.Series, s: pd.Series, boot: BootstrapConfig, rng: RngPair) -> float:
    rv = r.dropna().to_numpy(dtype=float)
    sv = s.dropna().to_numpy(dtype=float)
    if rv.size == 0 or sv.size == 0:
        return np.nan
    not_sig = 0
    for _ in range(boot.B):
        rb = rv[rng._rng.integers(0, rv.size, size=boot.sample_n)]
        sb = sv[rng._rng.integers(0, sv.size, size=boot.sample_n)]
        try:
            _, p_ks = ks_2samp(rb, sb)
        except ValueError:
            continue
        if p_ks > boot.alpha:
            not_sig += 1
    return not_sig / boot.B if boot.B else np.nan


def _shannon(series: pd.Series) -> float:
    v = series.dropna().value_counts(normalize=True)
    return float(entropy(v, base=2))


def run_t1(
    df_real: pd.DataFrame,
    df_sim: pd.DataFrame,
    variables: dict,
    boot: BootstrapConfig,
) -> dict:
    rng = RngPair()
    rows, entropy_rows = [], []
    for var, vcfg in variables.items():
        vtype = (vcfg.get("type") or "").lower()
        r = prep_variable(df_real, var, vcfg)
        s = prep_variable(df_sim, var, vcfg)
        if vtype == "categorical":
            rate = _bootstrap_cat(r, s, boot, rng)
        else:
            rate = _bootstrap_num(r, s, boot, rng)
        rows.append({"variable": var, "type": vtype, "insignificant_rate": rate})
        entropy_rows.append({"var": var, "real": _shannon(r), "sim": _shannon(s)})
        print(f"[t1] {var}: rate={rate:.3f}" if not np.isnan(rate) else f"[t1] {var}: rate=nan")

    summary = pd.DataFrame(rows)
    overall = float(np.nanmean(summary["insignificant_rate"])) if len(summary) else np.nan
    return {
        "avg_insignificant_rate": overall,
        "summary_df": summary,
        "extra": {"entropy": pd.DataFrame(entropy_rows)},
    }
