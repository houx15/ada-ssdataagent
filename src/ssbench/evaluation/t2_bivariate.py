"""T2 — pairwise association-strength comparison (Cramér's V / correlation / eta²).

Hot path works on pre-encoded numpy arrays with integer-index bootstrap draws;
statistical semantics identical to the pandas/autograd reference port.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ssbench.evaluation.bootstrap import BootstrapConfig, RngPair
from ssbench.evaluation.cleaning import prep_variable
from ssbench.evaluation.fast_stats import (
    corr_fisher_p,
    equal_assoc_cat_cat_codes,
    equal_assoc_num_cat_codes,
)


def _encode_union(*series) -> tuple[np.ndarray, ...]:
    """Encode series onto the sorted union of their values as int codes."""
    cats = sorted(set().union(*(set(pd.Series(s).dropna()) for s in series)))
    idx = {c: i for i, c in enumerate(cats)}
    return tuple(np.asarray([idx.get(v, -1) for v in s], dtype=np.int64) for s in series)


def run_t2(
    df_real: pd.DataFrame,
    df_sim: pd.DataFrame,
    variables: dict,
    boot: BootstrapConfig,
) -> dict:
    for df in (df_real, df_sim):
        for var, vcfg in variables.items():
            if var in df.columns:
                df[var] = prep_variable(df, var, vcfg)

    rng = RngPair()
    results = []
    var_list = list(variables)

    for i in range(len(var_list)):
        for j in range(i + 1, len(var_list)):
            v1, v2 = var_list[i], var_list[j]
            cfg1, cfg2 = variables[v1], variables[v2]
            if cfg1.get("input", False) and cfg2.get("input", False):
                continue

            t1 = (cfg1.get("type") or "categorical").lower()
            t2 = (cfg2.get("type") or "categorical").lower()

            df_r = df_real.dropna(subset=[v1, v2])
            df_s = df_sim.dropna(subset=[v1, v2])
            if df_r.empty or df_s.empty:
                rate = np.nan
            else:
                a1r = df_r[v1].to_numpy()
                a2r = df_r[v2].to_numpy()
                a1s = df_s[v1].to_numpy()
                a2s = df_s[v2].to_numpy()

                if t1 == "categorical" and t2 == "categorical":
                    x_r, x_s = _encode_union(a1r, a1s)
                    y_r, y_s = _encode_union(a2r, a2s)
                    nr = max(int(x_r.max()), int(x_s.max())) + 1
                    nc = max(int(y_r.max()), int(y_s.max())) + 1
                    n_r, n_s = len(x_r), len(x_s)

                    def one_draw(ri, si):
                        return equal_assoc_cat_cat_codes(
                            x_r[ri], y_r[ri], x_s[si], y_s[si], nr, nc,
                        )
                elif t1 == "numeric" and t2 == "numeric":
                    n_r, n_s = len(a1r), len(a1s)
                    a1r = a1r.astype(float)
                    a2r = a2r.astype(float)
                    a1s = a1s.astype(float)
                    a2s = a2s.astype(float)

                    def one_draw(ri, si):
                        return corr_fisher_p(
                            a1r[ri], a2r[ri], a1s[si], a2s[si],
                        )
                else:
                    num_r, cat_r_, num_s, cat_s_ = (
                        (a1r, a2r, a1s, a2s) if t1 == "numeric" else (a2r, a1r, a2s, a1s)
                    )
                    c_r, c_s = _encode_union(cat_r_, cat_s_)
                    num_r = num_r.astype(float)
                    num_s = num_s.astype(float)
                    k = max(int(c_r.max()), int(c_s.max())) + 1
                    n_r, n_s = len(num_r), len(num_s)

                    def one_draw(ri, si):
                        return equal_assoc_num_cat_codes(
                            num_r[ri], c_r[ri], num_s[si], c_s[si], k,
                        )

                wins = 0
                for _ in range(boot.B):
                    ri, si = rng.indices(n_r, n_s, boot.sample_n)
                    p = one_draw(ri, si)
                    if not np.isnan(p) and p > boot.alpha:
                        wins += 1
                rate = wins / boot.B

            results.append({
                "var1": v1, "var2": v2, "type1": t1, "type2": t2,
                "insignificant_rate": rate,
            })
            print(f"[t2] {v1} × {v2}: rate={rate:.3f}" if not np.isnan(rate)
                  else f"[t2] {v1} × {v2}: rate=nan")

    summary = pd.DataFrame(results)
    overall = float(np.nanmean(summary["insignificant_rate"])) if len(summary) else np.nan
    return {"avg_insignificant_rate": overall, "summary_df": summary}
