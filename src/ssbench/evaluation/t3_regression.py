"""T3 — multivariate regression comparison (R² strength via delta-method z-test)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from ssbench.evaluation.bootstrap import BootstrapConfig, RngPair
from ssbench.evaluation.cleaning import clean_regression_series
from ssbench.evaluation.stats_tests import equal_r2


def _fit_ols(df: pd.DataFrame, formula: str):
    try:
        return smf.ols(formula, data=df).fit()
    except Exception:  # noqa: BLE001
        return None


def _real_model_ok(df_real: pd.DataFrame, y: str, xs: list[str]) -> bool:
    formula = f"{y} ~ {' + '.join(xs)}"
    m = _fit_ols(df_real.dropna(subset=[y] + xs), formula)
    if m is None:
        return False
    if not hasattr(m, "params") or len(m.params) < 2 or np.all(pd.isna(m.params)):
        return False
    if hasattr(m, "df_resid") and m.df_resid <= 0:
        return False
    r2 = getattr(m, "rsquared", None)
    return r2 is not None and not np.isnan(r2)


def _clean_for_bootstrap(df: pd.DataFrame, y: str, xs: list[str]) -> pd.DataFrame:
    df = df.copy()
    df[y] = pd.to_numeric(df[y], errors="coerce")
    for x in xs:
        if x in df.columns:
            if pd.api.types.is_numeric_dtype(df[x]):
                df[x] = pd.to_numeric(df[x], errors="coerce")
            else:
                df[x] = (df[x].astype(str)
                         .replace({"": np.nan, "nan": np.nan, "NA": np.nan})
                         .astype("category"))
    return df


def run_t3(
    df_real: pd.DataFrame,
    df_sim: pd.DataFrame,
    responses: dict,
    predictors: dict,
    model_type,
    boot: BootstrapConfig,
) -> dict:
    for v, spec in predictors.items():
        for df in (df_real, df_sim):
            if v in df.columns:
                df[v] = clean_regression_series(df[v], spec)
    for y, spec in responses.items():
        for df in (df_real, df_sim):
            if y in df.columns:
                df[y] = clean_regression_series(df[y], spec)

    xs = list(predictors)
    response_names = list(responses)
    if isinstance(model_type, dict):
        pairs = [(y, model_type.get(y, "ols")) for y in response_names]
    elif isinstance(model_type, list):
        pairs = list(zip(response_names, model_type))
    else:
        pairs = [(y, model_type) for y in response_names]

    results = []
    for y, mt in pairs:
        if mt.lower() != "ols":
            print(f"[t3] skip {y}: unsupported model_type {mt}")
            continue
        if not _real_model_ok(df_real, y, xs):
            print(f"[t3] skip {y}: real model failed to fit")
            continue

        df_r = _clean_for_bootstrap(df_real, y, xs)
        df_s = _clean_for_bootstrap(df_sim, y, xs)
        formula = f"{y} ~ {' + '.join(xs)}"
        cols = [y] + [x for x in xs if x in df_r.columns]
        valid_r = df_r[cols].notna().all(axis=1).to_numpy()
        valid_s = df_s[cols].notna().all(axis=1).to_numpy()
        min_n = len(xs) * 2 + 2
        rng = RngPair(boot.seed)
        points, fit_fail = [], 0

        for _ in range(boot.B):
            ri, si = rng.indices(len(df_r), len(df_s), boot.sample_n)
            if int(valid_r[ri].sum() + valid_s[si].sum()) < min_n:
                points.append(0)
                continue
            rb = df_r.iloc[ri]
            sb = df_s.iloc[si]
            try:
                mr = smf.ols(formula, data=rb).fit()
                ms = smf.ols(formula, data=sb).fit()
                p = equal_r2(mr.rsquared, len(rb), ms.rsquared, len(sb))
            except Exception:  # noqa: BLE001
                fit_fail += 1
                p = np.nan
            points.append(1 if (not np.isnan(p) and p > boot.alpha) else 0)

        rate = float(np.mean(points)) if points else np.nan
        results.append({
            "response": y, "model_type": mt,
            "insignificant_rate": rate,
            "iterations": len(points), "pass_count": int(sum(points)),
            "fit_fail": fit_fail,
        })
        print(f"[t3] {y}: rate={rate:.3f}")

    summary = pd.DataFrame(results)
    overall = float(np.nanmean(summary["insignificant_rate"])) if len(summary) else np.nan
    return {"avg_insignificant_rate": overall, "summary_df": summary}
