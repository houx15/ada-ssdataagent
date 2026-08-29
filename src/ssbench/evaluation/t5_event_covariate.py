"""T5 — association strength between event order and covariates.

Hot path pre-encodes event-order labels and categorical predictors once per
combo/predictor, then draws bootstrap indices; statistics identical to the
pandas/autograd reference port.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from ssbench.evaluation.bootstrap import BootstrapConfig, RngPair
from ssbench.evaluation.cleaning import clean_event_series, prep_variable
from ssbench.evaluation.fast_stats import (
    equal_assoc_cat_cat_codes,
    equal_assoc_num_cat_codes,
)
from ssbench.evaluation.t2_bivariate import _encode_union
from ssbench.evaluation.t4_event_order import _order_label


def run_t5(
    df_real: pd.DataFrame,
    df_sim: pd.DataFrame,
    events: dict,
    predictors: dict,
    boot: BootstrapConfig,
) -> dict:
    for v, spec in events.items():
        df_real[v] = pd.to_numeric(clean_event_series(df_real[v], spec), errors="coerce")
        df_sim[v] = pd.to_numeric(clean_event_series(df_sim[v], spec), errors="coerce")
    for v, spec in predictors.items():
        for df in (df_real, df_sim):
            if v in df.columns:
                df[v] = prep_variable(df, v, spec)

    event_vars = list(events)
    combos = [list(c) for c in itertools.combinations(event_vars, 3)]
    rng = RngPair(boot.seed)
    results = []

    for combo in combos:
        r = df_real.dropna(subset=combo, how="any").copy()
        s = df_sim.dropna(subset=combo, how="any").copy()
        r["event_order"] = r.apply(_order_label, axis=1, args=(combo,), sep="-")
        s["event_order"] = s.apply(_order_label, axis=1, args=(combo,), sep="-")
        real_valid = r.dropna(subset=["event_order"])
        sim_valid = s.dropna(subset=["event_order"])

        if len(real_valid) < 5:
            print(f"[t5] skip combo {combo} (real < 5 valid samples)")
            continue

        for pred, pred_cfg in predictors.items():
            df_r = real_valid.dropna(subset=[pred, "event_order"])
            df_s = sim_valid.dropna(subset=[pred, "event_order"])
            if df_r.empty or df_s.empty:
                continue

            pred_type = (pred_cfg.get("type") or "categorical").lower()
            eo_r = df_r["event_order"].to_numpy()
            eo_s = df_s["event_order"].to_numpy()
            pv_r = df_r[pred].to_numpy()
            pv_s = df_s[pred].to_numpy()
            n_r, n_s = len(df_r), len(df_s)

            if pred_type == "categorical":
                x_r, x_s = _encode_union(eo_r, eo_s)
                y_r, y_s = _encode_union(pv_r, pv_s)
                nr = max(int(x_r.max()), int(x_s.max())) + 1
                nc = max(int(y_r.max()), int(y_s.max())) + 1

                def one_draw(ri, si):
                    return equal_assoc_cat_cat_codes(
                        x_r[ri], y_r[ri], x_s[si], y_s[si], nr, nc,
                    )
            else:
                x_r, x_s = _encode_union(eo_r, eo_s)
                num_r = pv_r.astype(float)
                num_s = pv_s.astype(float)
                k = max(int(x_r.max()), int(x_s.max())) + 1

                def one_draw(ri, si):
                    return equal_assoc_num_cat_codes(
                        num_r[ri], x_r[ri], num_s[si], x_s[si], k,
                    )

            pvals = []
            for _ in range(boot.B):
                try:
                    ri, si = rng.indices(n_r, n_s, boot.sample_n)
                    p = one_draw(ri, si)
                    pvals.append(0 if np.isnan(p) else p)
                except Exception:  # noqa: BLE001
                    pvals.append(0)

            if not pvals:
                continue
            rate = float(np.mean(np.array(pvals) > boot.alpha))
            results.append({
                "combo": "→".join(combo),
                "predictor": pred,
                "type": pred_type,
                "insignificant_rate": rate,
                "iterations": len(pvals),
            })
            print(f"[t5] {'→'.join(combo)} × {pred}: rate={rate:.3f}")

    summary = pd.DataFrame(results)
    overall = float(np.nanmean(summary["insignificant_rate"])) if len(summary) else np.nan
    return {"avg_insignificant_rate": overall, "summary_df": summary}
