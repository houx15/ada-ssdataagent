#!/usr/bin/env python3
"""Analyze within-CFPS ADA self-correction (no real data in stage calls).

Compares, per T1 field:
  raw Q                       (Actor empirical distribution)
  pure arbiter softmax(phi)   (direct ADA replacement)
  x_Q + beta * r_ADA          (innovation correction; beta leave-one-field-out
                               within CFPS, plus per-field oracle upper bound)
Metrics: TV for categorical / binned fields; KS sup error via reconstructed
CDF for numeric fields.

Usage:
  .venv/bin/python scripts/ada_cfps_analyze.py --dir runs/ada/cfps_<stamp>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ssbench.ada.signals import aggregate_edges, clr_with_pseudo, hodge, softmax  # noqa: E402
from ssbench.evaluation.cleaning import prep_variable  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUN = os.path.join(ROOT, "runs", "cfps", "direct", "20260817_164329_glm52")
DELTA = 0.5


def tv(p, q):
    return float(0.5 * np.abs(p - q).sum())


def load_units(path):
    units = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                units.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return units


def field_signals(units, field):
    """Aggregate per-(field, edge) responses across personas into Hodge potential."""
    m_labels = None
    shim = []
    for u in units:
        edges, g = [], {}
        for e in u.get("edges", []):
            if e["field"] == field:
                edges.append({"edge": tuple(e["edge"]), "source": e["source"]})
        for k, v in u.get("g", {}).items():
            var, jk = k.split("|")
            if var == field:
                j, kk = jk.split(",")
                g[(int(j), int(kk))] = v
        if g:
            shim.append({"edges": edges, "g": g})

    class R:
        pass

    objs = []
    for s in shim:
        r = R()
        r.edges = s["edges"]
        r.g = s["g"]
        objs.append(r)
    if not objs:
        return None
    n_levels = max(max(max(e["edge"]) for e in o.edges) for o in objs) + 1
    gbar, counts, src_g = aggregate_edges(objs, ["x"] * n_levels)
    return {"n_units": len(objs), "gbar": gbar, "counts": counts, "src_g": src_g}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    d = args.dir

    with open(os.path.join(d, "levels.json"), encoding="utf-8") as f:
        levels = json.load(f)
    units = load_units(os.path.join(d, "units.jsonl"))

    with open(os.path.join(ROOT, "configs", "eval", "cfps.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    real = pd.read_csv(os.path.join(RUN, "real.csv"), low_memory=False)
    sim = pd.read_csv(os.path.join(RUN, "sim.csv"), low_memory=False)

    fields = []
    for var, lm in levels.items():
        vcfg = dict(cfg["t1"]["variables"][var])
        labels = lm["labels"]
        kind = lm["kind"]
        meta = lm.get("meta", {})

        def to_idx(series):
            out = []
            for v in series:
                if pd.isna(v):
                    out.append(-1)
                elif kind == "categorical":
                    out.append(labels.index(v) if v in labels else -1)
                elif kind == "numeric_unique":
                    arr = np.asarray(labels, dtype=float)
                    j = int(np.argmin(np.abs(arr - float(v))))
                    out.append(j if abs(arr[j] - float(v)) < 1e-9 else -1)
                else:
                    edges = meta["edges"]
                    full = [-np.inf] + list(edges) + [np.inf]
                    j = None
                    for i in range(len(labels)):
                        if full[i] < float(v) <= full[i + 1]:
                            j = i
                            break
                    out.append(j if j is not None else -1)
            return np.asarray(out, dtype=int)

        ri = to_idx(prep_variable(real, var, vcfg).to_numpy())
        si = to_idx(prep_variable(sim, var, vcfg).to_numpy())
        m = len(labels)
        pc = np.bincount(ri[ri >= 0], minlength=m).astype(float)
        qc = np.bincount(si[si >= 0], minlength=m).astype(float)
        if (pc.sum() < 50) or (qc.sum() < 50) or len(labels) < 2:
            continue
        fields.append({"var": var, "labels": labels, "kind": kind, "meta": meta,
                       "pc": pc, "qc": qc, "vcfg": vcfg,
                       "sim_vals": prep_variable(sim, var, vcfg).dropna().to_numpy(float)
                       if kind.startswith("numeric") else None,
                       "real_vals": prep_variable(real, var, vcfg).dropna().to_numpy(float)
                       if kind.startswith("numeric") else None})

    sig = {}
    for f in fields:
        s = field_signals(units, f["var"])
        if s and len(s["gbar"]) >= 1:
            phi, resid = hodge(s["gbar"], f["labels"])
            s["phi"] = phi
            s["resid"] = resid
            sig[f["var"]] = s

    # ---- pure-phi replacement + beta innovation ----
    def dist_from_x(x, f):
        p = softmax(x)
        return p

    # beta fitted by leaving the field out (within CFPS)
    def fit_beta(exclude):
        num = den = 0.0
        for var, s in sig.items():
            if var == exclude:
                continue
            f = next(x for x in fields if x["var"] == var)
            x_q = clr_with_pseudo(f["qc"])
            x_p = clr_with_pseudo(f["pc"])
            r = s["phi"] - x_q
            num += float(r @ (x_p - x_q))
            den += float(r @ r)
        return num / den if den > 0 else 0.0

    betas = {var: fit_beta(var) for var in sig}

    rows = []
    for f in fields:
        var = f["var"]
        if var not in sig:
            continue
        s = sig[var]
        p_hat_counts = (f["pc"] + DELTA)
        p = f["pc"] / f["pc"].sum()
        q = f["qc"] / f["qc"].sum()
        x_q = clr_with_pseudo(f["qc"])
        x_p = clr_with_pseudo(f["pc"])
        phi = s["phi"]
        p_phi = softmax(phi)
        beta = betas[var]
        r_ada = phi - x_q
        p_beta = softmax(x_q + beta * r_ada)
        # oracle beta on this field (upper bound, uses real P - reported only)
        denom = float(r_ada @ r_ada)
        beta_or = float(r_ada @ (x_p - x_q)) / denom if denom > 0 else 0.0
        p_or = softmax(x_q + beta_or * r_ada)

        row = {"var": var, "kind": f["kind"], "m": len(f["labels"]),
               "n_units": s["n_units"], "n_edges": len(s["gbar"]),
               "tv_raw": tv(p, q), "tv_phi": tv(p, p_phi),
               "tv_beta_lofo": tv(p, p_beta), "beta_lofo": beta,
               "tv_beta_oracle": tv(p, p_or), "beta_oracle": beta_or,
               "cos_innov": float((r_ada @ (x_p - x_q)) /
                                  (np.linalg.norm(r_ada) * np.linalg.norm(x_p - x_q) + 1e-12))}
        if f["kind"].startswith("numeric") and f["real_vals"] is not None:
            row["ks_raw"] = ks_from_bins(q, f)
            row["ks_beta"] = ks_from_bins(p_beta, f)
            row["ks_phi"] = ks_from_bins(p_phi, f)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(d, "analysis.csv"), index=False)
    cols = [c for c in ["var", "kind", "m", "n_units", "n_edges", "tv_raw", "tv_phi",
                        "tv_beta_lofo", "beta_lofo", "tv_beta_oracle", "beta_oracle",
                        "cos_innov", "ks_raw", "ks_phi", "ks_beta"] if c in df.columns]
    print(df[cols].round(3).to_string(index=False))

    print("\n===== summary (means) =====")
    summ = {}
    cat = df[df["kind"] == "categorical"]
    if len(cat):
        summ["categorical (TV)"] = {
            "n": len(cat), "raw": cat.tv_raw.mean(), "pure_phi": cat.tv_phi.mean(),
            "beta_lofo": cat.tv_beta_lofo.mean(), "beta_oracle": cat.tv_beta_oracle.mean(),
            "win_lofo": float((cat.tv_beta_lofo < cat.tv_raw).mean()),
            "win_phi": float((cat.tv_phi < cat.tv_raw).mean())}
    num = df[df["kind"].str.startswith("numeric")]
    if len(num):
        summ["numeric (KS)"] = {
            "n": len(num), "raw": num.ks_raw.mean(), "pure_phi": num.ks_phi.mean(),
            "beta_lofo": num.ks_beta.mean(),
            "win_lofo": float((num.ks_beta < num.ks_raw).mean()),
            "win_phi": float((num.ks_phi < num.ks_raw).mean())}
    print(json.dumps(summ, indent=2))
    with open(os.path.join(d, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summ, "betas": betas}, f, indent=2)
    print(f"\n[ada-analyze] -> {d}/analysis.csv")


def ks_from_bins(bin_probs, f):
    """KS sup error of the reconstructed CDF vs the real sample.

    Within each bin the sim empirical CDF is rescaled by the bin probability."""
    labels = f["labels"]
    kind = f["kind"]
    meta = f["meta"]
    real_vals = np.sort(f["real_vals"])
    sim_vals = f["sim_vals"]
    if kind == "numeric_unique":
        uniq = np.asarray(labels, dtype=float)
        edges = [(u - 0.5, u + 0.5) for u in uniq]
    else:
        e = meta["edges"]
        edges = [(-np.inf, e[0])] + [(e[i], e[i + 1]) for i in range(len(e) - 1)] + [(e[-1], np.inf)]
    cdf_pts = np.concatenate([[0.0], np.cumsum(bin_probs)])
    grid = real_vals
    out = np.empty(len(grid))
    for i, v in enumerate(grid):
        b = None
        for j, (lo, hi) in enumerate(edges):
            if lo < v <= hi or (j == 0 and v <= hi) or (j == len(edges) - 1 and v > lo):
                b = j
                break
        if b is None:
            b = len(edges) - 1
        base = cdf_pts[b]
        in_bin = sim_vals[(sim_vals > edges[b][0]) & (sim_vals <= edges[b][1])]
        if len(in_bin) > 1:
            frac = (np.searchsorted(np.sort(in_bin), v, side="right") / len(in_bin))
        else:
            frac = 1.0 if (len(in_bin) and v >= in_bin[0]) else 0.0
        out[i] = base + bin_probs[b] * frac
    f_real = np.arange(1, len(grid) + 1) / len(grid)
    return float(np.abs(out - f_real).max())


if __name__ == "__main__":
    main()
