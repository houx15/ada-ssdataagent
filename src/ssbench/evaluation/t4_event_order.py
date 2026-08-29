"""T4 — life-event order comparison (chi-square over order-pattern distributions)."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, entropy

from ssbench.evaluation.bootstrap import BootstrapConfig, RngPair
from ssbench.evaluation.cleaning import clean_event_series


def _order_label(row, event_vars: list[str], sep: str = "→") -> object:
    events = []
    for v in event_vars:
        val = row[v]
        if pd.isna(val):
            continue
        events.append((v, float(val)))
    if not events:
        return np.nan
    ordered = sorted(events, key=lambda x: x[1])
    return sep.join([v for v, _ in ordered])


def _shannon_entropy(series: pd.Series) -> float:
    v = series.dropna().value_counts(normalize=True)
    return entropy(v, base=2)


def run_t4(
    df_real: pd.DataFrame,
    df_sim: pd.DataFrame,
    events: dict,
    boot: BootstrapConfig,
) -> dict:
    event_vars = list(events)
    for v, spec in events.items():
        df_real[v] = clean_event_series(df_real[v], spec)
        df_sim[v] = clean_event_series(df_sim[v], spec)

    combos = [list(c) for c in itertools.combinations(event_vars, 3)]
    rng = RngPair(boot.seed)
    results, entropy_rows = [], []

    for combo in combos:
        r = df_real.dropna(subset=combo, how="any").copy()
        s = df_sim.dropna(subset=combo, how="any").copy()
        r["event_order"] = r.apply(_order_label, axis=1, args=(combo,))
        s["event_order"] = s.apply(_order_label, axis=1, args=(combo,))

        entropy_rows.append({
            "combo": "→".join(combo),
            "real_entropy": _shannon_entropy(r["event_order"]),
            "sim_entropy": _shannon_entropy(s["event_order"]),
        })

        pvals, dissim = [], []
        for _ in range(boot.B):
            try:
                rb, sb = rng.draw(r, s, boot.sample_n)
                all_orders = sorted(set(rb["event_order"]) | set(sb["event_order"]))
                if not all_orders:
                    continue
                tbl = pd.DataFrame(0, index=["real", "sim"], columns=all_orders)
                tbl.loc["real"] = rb["event_order"].value_counts().reindex(all_orders, fill_value=0)
                tbl.loc["sim"] = sb["event_order"].value_counts().reindex(all_orders, fill_value=0)
                p_real = tbl.loc["real"] / tbl.loc["real"].sum()
                p_sim = tbl.loc["sim"] / tbl.loc["sim"].sum()
                dissim.append(0.5 * float(np.sum(np.abs(p_real - p_sim))))
                _, p, _, _ = chi2_contingency(tbl, correction=False)
                pvals.append(0 if np.isnan(p) else p)
            except Exception:  # noqa: BLE001
                continue

        if not pvals:
            res = {"insignificant_rate": np.nan, "mean_dissimilarity": np.nan,
                   "pass_count": 0, "iterations": 0}
        else:
            non_sig = int(np.sum(np.array(pvals) >= boot.alpha))
            res = {
                "insignificant_rate": non_sig / len(pvals),
                "mean_p": float(np.mean(pvals)),
                "mean_dissimilarity": float(np.mean(dissim)) if dissim else np.nan,
                "pass_count": non_sig,
                "iterations": len(pvals),
            }
        res["combo"] = "→".join(combo)
        results.append(res)
        rate = res["insignificant_rate"]
        print(f"[t4] {'→'.join(combo)}: rate={rate:.3f}" if not np.isnan(rate)
              else f"[t4] {'→'.join(combo)}: rate=nan")

    summary = pd.DataFrame(results)
    overall = float(np.nanmean(summary["insignificant_rate"])) if len(summary) else np.nan
    return {
        "avg_insignificant_rate": overall,
        "summary_df": summary,
        "extra": {"entropy": pd.DataFrame(entropy_rows)},
    }
