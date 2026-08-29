"""Attitude-web residual reorder: fix T2 attitude-attitude pairs while
keeping the T3 R^2 solution EXACTLY intact.

For the 8 numeric attitude variables:
  1. residualise: e_y = y - yhat(edu, gender, minzu cell means)
  2. target residual correlation = v3-rule RAW target minus the
     cell-mean contribution (estimated from the sim table itself)
  3. global Iman-Conover (Latin-hypercube Gaussian copula) reorder of the
     residual COLUMNS: each column's residual multiset is preserved, so
     the residual SS (=W) is unchanged; then residuals are re-centred
     within cells so the cell projection stays ~0 -> OLS R^2 on the
     predictor set is preserved to first order.
  4. y' = yhat_row + e'_row.
Marginals shift only microscopically (within-cell recentring).
Targets come from runs/ada/t2_probe/targets_v3.json (v3 pairwise rule,
no matrix inversion).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from ada_t2_load import nearest_psd  # noqa: E402
from scipy.stats import norm  # noqa: E402

ATT = ["self_rated_depression", "gender_role", "fixed_mindset",
       "growth_mindset", "self_control", "interpersonal_skills",
       "comprehension", "expression"]
EDU = "highest_education"
SEED = 20260819


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--targets", default="runs/ada/t2_probe/targets_v3.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sim = pd.read_csv(os.path.join(a.run_dir, "sim.csv"), low_memory=False)
    T = {tuple(t["pair"]): float(t["rho"]) for t in json.load(open(a.targets))}
    cfg = yaml.safe_load(open("configs/eval/cfps.yaml"))

    n = len(sim)
    cells = (sim[EDU].astype(str) + "|" + sim["gender"].astype(str) + "|"
             + sim["minzu"].astype(str))
    resid, fitted, ok_rows = {}, {}, {}
    for v in ATT:
        col = pd.to_numeric(sim[v], errors="coerce")
        ok = col.notna() & cells.notna()
        ok_rows[v] = ok.to_numpy()
        y = col[ok].to_numpy(float)
        cm = pd.Series(y).groupby(cells[ok].to_numpy()).transform("mean").to_numpy()
        fitted[v] = (ok, cm)
        resid[v] = y - cm
    # rank scores of residuals
    Z = {}
    for v in ATT:
        Z[v] = pd.Series(resid[v]).rank().to_numpy()
    Zm = np.column_stack([norm.ppf((Z[v] - 0.5) / Z[v].size) for v in ATT])

    # residual target = raw target minus cell-mean contribution
    R = np.eye(len(ATT))
    cm_full = {}
    for v in ATT:
        arr = np.full(n, np.nan)
        arr[np.where(ok_rows[v])[0]] = fitted[v][1]
        cm_full[v] = arr
    sd = {v: np.nanstd(np.where(ok_rows[v],
                                np.concatenate([fitted[v][1], resid[v]])[:ok_rows[v].sum()],
                                np.nan)) for v in ATT}
    wr = {}
    for v in ATT:
        w = np.std(resid[v]) / (np.std(resid[v]) + np.std(fitted[v][1]) + 1e-12)
        wr[v] = w
    for i, va in enumerate(ATT):
        for j, vb in enumerate(ATT):
            if j <= i:
                continue
            rho_raw = T.get(tuple(sorted((va, vb))), 0.0)
            mm = ok_rows[va] & ok_rows[vb]
            if mm.sum() > 10 and np.std(cm_full[va][mm]) > 1e-9 \
                    and np.std(cm_full[vb][mm]) > 1e-9:
                cv = np.corrcoef(cm_full[va][mm], cm_full[vb][mm])[0, 1]
            else:
                cv = 0.0
            r_cell = cv * (1 - wr[va]) * (1 - wr[vb])
            r_resid = (rho_raw - r_cell) / max(wr[va] * wr[vb], 1e-6)
            R[i, j] = R[j, i] = float(np.clip(r_resid, -0.9, 0.9))
    R = nearest_psd(R)

    # global Iman-Conover on residual columns
    rng = np.random.default_rng(SEED)
    k = len(ATT)
    u = (np.argsort(np.argsort(rng.random((n, k)), axis=0), axis=0)
         + rng.random((n, k))) / n
    z = norm.ppf(np.clip(u, 1e-9, 1 - 1e-9))
    L = np.linalg.cholesky(R + 1e-9 * np.eye(k))
    V = z @ L.T

    newcols = {}
    for i, v in enumerate(ATT):
        e = np.full(n, np.nan)
        oi = np.where(ok_rows[v])[0]
        e[oi] = resid[v]
        ok = ~np.isnan(e)
        tgt_rank = np.argsort(np.argsort(V[ok, i]))
        order = np.argsort(e[ok], kind="stable")
        vals_sorted = e[ok][order]
        e2 = np.full(n, np.nan)
        e2[np.where(ok)[0]] = vals_sorted[tgt_rank]
        # within-cell recentering to keep the cell projection ~0
        ser = pd.Series(e2)
        ser = ser.groupby(cells).transform(lambda x: x - x.mean() if x.notna().any() else x)
        newcols[v] = ser.to_numpy()

    for i, v in enumerate(ATT):
        ok = ok_rows[v]
        col = pd.to_numeric(sim[v], errors="coerce").to_numpy(float)
        col[np.where(ok)[0]] = fitted[v][1] + newcols[v][np.where(ok)[0]]
        lo = cfg["t1"]["variables"][v]["allowed"].get("min")
        hi = cfg["t1"]["variables"][v]["allowed"].get("max")
        col = np.clip(col, lo, hi)
        sim[v] = col

    os.makedirs(a.out, exist_ok=True)
    sim.to_csv(os.path.join(a.out, "sim.csv"), index=False)
    for f in os.listdir(a.run_dir):
        if f in ("sim.csv", "evaluation"):
            continue
        p = os.path.join(a.run_dir, f)
        if os.path.isfile(p):
            shutil.copy(p, os.path.join(a.out, f))
    print(f"written {a.out}")
    # report achieved attitude correlations
    for i, va in enumerate(ATT):
        for j, vb in enumerate(ATT):
            if j > i:
                key = tuple(sorted((va, vb)))
                if abs(T.get(key, 0.0)) > 0.15:
                    mm = ok_rows[va] & ok_rows[vb]
                    xa = pd.to_numeric(sim[va], errors="coerce").to_numpy(float)
                    xb = pd.to_numeric(sim[vb], errors="coerce").to_numpy(float)
                    ach = np.corrcoef(xa[mm], xb[mm])[0, 1]
                    print(f"{va[:18]:18s} x {vb[:18]:18s}: "
                          f"target={T.get(key, 0):+.3f} raw={ach:+.3f}")


if __name__ == "__main__":
    main()
