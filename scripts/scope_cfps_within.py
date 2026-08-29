"""Within-CFPS cross-field equalizer transfer (leave-one-field-out detail).

Questions:
1. Can CFPS's own other fields predict each field's channel distortion (LOO)?
2. Does adding GSS blocks to training help within-CFPS prediction?
3. Same for association blocks (LOO over variable pairs within CFPS).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import scope_phase_a as pa  # noqa: E402  (reuse its builders)

ROOT = pa.ROOT
OUT = os.path.join(ROOT, "analysis_out", "scope_phase_a")
LAM = 0.1


def main():
    cfgs = {ds: pa.load_eval_cfg(ds) for ds in pa.RUNS}
    frames = {ds: pa.load_frames(ds) for ds in pa.RUNS}
    cat_blocks = {ds: pa.build_cat_blocks(ds, *frames[ds], cfgs[ds]) for ds in pa.RUNS}
    num_blocks = {ds: pa.build_num_blocks(ds, *frames[ds], cfgs[ds]) for ds in pa.RUNS}
    all_assoc = {ds: pd.DataFrame(pa.assoc_blocks(ds, *frames[ds], cfgs[ds])) for ds in pa.RUNS}

    rows = []
    for kind in ("ordinal", "nominal"):
        keys = [k for k in cat_blocks["cfps"] if k[0] == kind]
        for tk in keys:
            b = cat_blocks["cfps"][tk]
            fn = pa.equalize_cat if kind == "ordinal" else pa.equalize_nominal
            # within-CFPS LOO
            tr_c = [k for k in keys if k != tk]
            p_hat_c, d_hat = fn(cat_blocks["cfps"], tr_c, b, LAM, return_vec=True)
            # +GSS same-kind blocks
            tr_cg = tr_c + [k for k in cat_blocks["gss"] if k[0] == kind]
            p_hat_cg, d_hat_g = fn({**cat_blocks["cfps"], **cat_blocks["gss"]},
                                   tr_cg, b, LAM, return_vec=True)
            # +GSS cross-kind (all graph blocks)
            tr_all = tr_cg + [k for k in cat_blocks["gss"] if k[0] != kind]
            p_hat_all, _ = fn({**cat_blocks["cfps"], **cat_blocks["gss"]},
                              tr_all, b, LAM, return_vec=True)
            d_vec = b["x"] - b["y"]
            rows.append({
                "kind": kind, "var": tk[1], "m": len(b["p"]),
                "tv_raw": pa.tv(b["p"], b["q"]),
                "tv_loo": pa.tv(b["p"], p_hat_c),
                "tv_gss": pa.tv(b["p"], p_hat_cg),
                "tv_gssall": pa.tv(b["p"], p_hat_all),
                "r2_loo": 1 - np.linalg.norm(d_vec - d_hat) ** 2 / np.linalg.norm(d_vec) ** 2,
                "r2_gss": 1 - np.linalg.norm(d_vec - d_hat_g) ** 2 / np.linalg.norm(d_vec) ** 2,
            })
    for tk, b in num_blocks["cfps"].items():
        tr_c = [k for k in num_blocks["cfps"] if k != tk]
        pred_c, d_hat = pa.equalize_num(num_blocks["cfps"], tr_c, b, LAM, return_vec=True)
        tr_cg = tr_c + list(num_blocks["gss"])
        merged = {**num_blocks["cfps"], **{("g__" + k): v for k, v in num_blocks["gss"].items()}}
        tr_cg = tr_c + [("g__" + k) for k in num_blocks["gss"]]
        pred_g, d_hat_g = pa.equalize_num(merged, tr_cg, b, LAM, return_vec=True)
        d_vec = b["x"] - b["y"]
        rows.append({
            "kind": "numeric", "var": tk, "m": len(pa.U_GRID),
            "tv_raw": pa.ecdf_sup_error(pa.quantiles(b["sim"]), b["real"]),
            "tv_loo": pa.ecdf_sup_error(pred_c, b["real"]),
            "tv_gss": pa.ecdf_sup_error(pred_g, b["real"]),
            "tv_gssall": np.nan,
            "r2_loo": 1 - np.linalg.norm(d_vec - d_hat) ** 2 / np.linalg.norm(d_vec) ** 2,
            "r2_gss": 1 - np.linalg.norm(d_vec - d_hat_g) ** 2 / np.linalg.norm(d_vec) ** 2,
        })
    t1_df = pd.DataFrame(rows)
    t1_df.to_csv(os.path.join(OUT, "cfps_within_t1.csv"), index=False)

    arows = []
    for atype in ("cat-cat", "num-cat", "num-num"):
        sub = all_assoc["cfps"][all_assoc["cfps"]["type"] == atype].reset_index(drop=True)
        sub_g = all_assoc["gss"][all_assoc["gss"]["type"] == atype]
        if len(sub) < 4:
            continue
        for i in range(len(sub)):
            row = sub.iloc[i]
            tr = sub.drop(i)
            zq, zp = pa.logit01(np.abs(tr["V_Q"].to_numpy())), pa.logit01(np.abs(tr["V_P"].to_numpy()))
            beta = np.linalg.lstsq(np.column_stack([np.ones(len(zq)), zq]), zp, rcond=None)[0]
            pred_loo = pa.inv_logit01(beta[0] + beta[1] * pa.logit01(abs(row["V_Q"])))
            tr2 = pd.concat([tr[["V_Q", "V_P"]], sub_g[["V_Q", "V_P"]]])
            zq2, zp2 = pa.logit01(np.abs(tr2["V_Q"].to_numpy())), pa.logit01(np.abs(tr2["V_P"].to_numpy()))
            beta2 = np.linalg.lstsq(np.column_stack([np.ones(len(zq2)), zq2]), zp2, rcond=None)[0]
            pred_gss = pa.inv_logit01(beta2[0] + beta2[1] * pa.logit01(abs(row["V_Q"])))
            arows.append({
                "type": atype, "var1": row["var1"], "var2": row["var2"],
                "err_raw": abs(row["V_Q"] - row["V_P"]),
                "err_loo": abs(pred_loo - row["V_P"]),
                "err_gss": abs(pred_gss - row["V_P"]),
            })
    assoc_df = pd.DataFrame(arows)
    assoc_df.to_csv(os.path.join(OUT, "cfps_within_assoc.csv"), index=False)

    print("=" * 78)
    print("CFPS 内部逐字段 LOO（tv/ks 越低越好；r2 为 clr/分位数残差解释率）")
    print(t1_df.round(3).to_string(index=False))
    print("\n汇总（CFPS 内部 LOO vs +GSS 混合训练）:")
    summ = t1_df.groupby("kind").agg(
        n=("tv_raw", "size"), raw=("tv_raw", "mean"), loo=("tv_loo", "mean"),
        gss=("tv_gss", "mean"), gssall=("tv_gssall", "mean"),
        r2_loo=("r2_loo", "mean"), r2_gss=("r2_gss", "mean"),
        win_loo=("tv_loo", lambda s: (s < t1_df.loc[s.index, "tv_raw"]).mean()),
        gss_beats=("tv_gss", lambda s: (s < t1_df.loc[s.index, "tv_loo"]).mean()),
    )
    print(summ.round(3).to_string())
    print("\n关联块 CFPS 内部 LOO（|V − V_P|，越低越好）:")
    if not assoc_df.empty:
        asum = assoc_df.groupby("type").agg(
            n=("err_raw", "size"), raw=("err_raw", "mean"),
            loo=("err_loo", "mean"), gss=("err_gss", "mean"),
            win_loo=("err_loo", lambda s: (s < assoc_df.loc[s.index, "err_raw"]).mean()),
            gss_beats=("err_gss", lambda s: (s < assoc_df.loc[s.index, "err_loo"]).mean()))
        print(asum.round(3).to_string())
        worst = assoc_df[assoc_df["err_loo"] > assoc_df["err_raw"]]
        print(f"\nLOO 变差的关联块 ({len(worst)}/{len(assoc_df)}):")
        print(worst.sort_values("err_loo", ascending=False).head(8).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
