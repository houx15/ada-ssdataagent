#!/usr/bin/env python3
"""Turn ADA field corrections into an evaluable CFPS population.

Per field: p_hat = softmax(x_q + beta * (phi - x_q)); then rank-preserving
marginal reallocation (individuals sorted by current value are reassigned to
target bins in order; within-bin values drawn from that bin's sim values at
evenly spaced quantiles). Preserves each field's per-person ranks, hence
cross-field rank associations.

Outputs sim_lofo.csv (per-field leave-one-field-out beta) and sim_oracle.csv
(per-field oracle beta; upper bound) next to the collect dir, plus shadow run
dirs with meta.json/real.csv/sim.csv for the official evaluator.

Usage:
  uv run python scripts/ada_make_sim.py --dir runs/ada/cfps_20260817_222857
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

from ssbench.ada.signals import clr_with_pseudo, hodge, softmax  # noqa: E402
from ssbench.evaluation.cleaning import prep_variable  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUN = os.path.join(ROOT, "runs", "cfps", "direct", "20260817_164329_glm52")


def field_g_phi(units, var, m):
    gacc = {}
    for u in units:
        for k, v in u["g"].items():
            va, jk = k.split("|")
            if va != var:
                continue
            j, kk = map(int, jk.split(","))
            gacc.setdefault((j, kk), []).append(v)
    if not gacc:
        return None, None
    gbar = {k: float(np.mean(v)) for k, v in gacc.items()}
    phi, _ = hodge(gbar, ["x"] * m)
    return gbar, phi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--run-dir", default=RUN,
                    help="direct seed run containing sim.csv/real.csv")
    ap.add_argument("--beta-modes", default="one,lofo,oracle",
                    help="comma list of: one (pure phi, zero hyperparameter), "
                         "lofo (leave-one-field-out beta), oracle (per-field upper bound)")
    args = ap.parse_args()
    d = args.dir
    run_dir = os.path.abspath(args.run_dir)
    args.beta_modes = [m.strip() for m in args.beta_modes.split(",") if m.strip()]

    levels = json.load(open(os.path.join(d, "levels.json"), encoding="utf-8"))
    units = [json.loads(l) for l in open(os.path.join(d, "units.jsonl"), encoding="utf-8")]
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "eval", "cfps.yaml"), encoding="utf-8"))
    real = pd.read_csv(os.path.join(run_dir, "real.csv"), low_memory=False)
    sim = pd.read_csv(os.path.join(run_dir, "sim.csv"), low_memory=False).copy()

    # per-field signals & betas
    info = {}
    for var, lm in levels.items():
        if lm.get("kind") == "order" or var == "__order__":
            continue  # order pseudo-field handled by ada_make_sim_order.py
        labels, kind, meta = lm["labels"], lm["kind"], lm.get("meta", {})
        m = len(labels)
        vcfg = dict(cfg["t1"]["variables"][var])
        _, phi = field_g_phi(units, var, m)
        if phi is None:
            continue

        def to_idx(values):
            out = []
            for v in values:
                if pd.isna(v):
                    out.append(-1)
                elif kind == "categorical":
                    out.append(labels.index(v) if v in labels else -1)
                elif kind == "numeric_unique":
                    arr = np.asarray(labels, float)
                    j = int(np.argmin(np.abs(arr - float(v))))
                    out.append(j if abs(arr[j] - float(v)) < 1e-9 else -1)
                else:
                    full = [-np.inf] + list(meta["edges"]) + [np.inf]
                    j = None
                    for i in range(len(labels)):
                        if full[i] < float(v) <= full[i + 1]:
                            j = i
                            break
                    out.append(j if j is not None else -1)
            return np.asarray(out, int)

        si = to_idx(prep_variable(sim, var, vcfg).to_numpy())
        ri = to_idx(prep_variable(real, var, vcfg).to_numpy())
        qc = np.bincount(si[si >= 0], minlength=m).astype(float)
        pc = np.bincount(ri[ri >= 0], minlength=m).astype(float)
        info[var] = {"labels": labels, "kind": kind, "meta": meta, "m": m,
                     "si": si, "qc": qc, "pc": pc, "phi": phi,
                     "x_q": clr_with_pseudo(qc), "x_p": clr_with_pseudo(pc)}

    def beta_for(var, mode):
        if mode == "one":
            return 1.0  # pure phi replacement: softmax(x_q + (phi - x_q)) = softmax(phi)
        r = info[var]["phi"] - info[var]["x_q"]
        if mode == "oracle":
            den = float(r @ r)
            return float(r @ (info[var]["x_p"] - info[var]["x_q"])) / den if den > 0 else 0.0
        num = den = 0.0
        for v2, s2 in info.items():
            if v2 == var:
                continue
            r2 = s2["phi"] - s2["x_q"]
            num += float(r2 @ (s2["x_p"] - s2["x_q"]))
            den += float(r2 @ r2)
        return num / den if den > 0 else 0.0

    def reallocate(var, beta):
        """Rank-preserving marginal reallocation on sim[var]."""
        s = info[var]
        labels, kind, meta, m = s["labels"], s["kind"], s["meta"], s["m"]
        si = s["si"]
        r_ada = s["phi"] - s["x_q"]
        p_hat = softmax(s["x_q"] + beta * r_ada)
        n = (si >= 0).sum()
        target = np.floor(p_hat * n).astype(int)
        # largest-remainder fill to n
        rem = n - target.sum()
        order = np.argsort(-(p_hat * n - target))
        for b in range(rem):
            target[order[b % m]] += 1
        # per-bin source values from sim
        vcfg = dict(cfg["t1"]["variables"][var])
        vals = prep_variable(sim, var, vcfg).to_numpy()
        bin_vals = {b: np.sort(vals[si == b].astype(float)) if kind != "categorical"
                    else None for b in range(m)}
        out = np.full(len(vals), np.nan, dtype=object)
        # individuals sorted by current bin (stable by original value order)
        idx_sorted = np.argsort(np.where(si >= 0, si, 10**9), kind="stable")
        slot = np.concatenate([[b] * target[b] for b in range(m)]) if target.sum() else []
        slot = np.asarray(slot, int)
        assert len(slot) == n
        cursor = {b: 0 for b in range(m)}
        rng = np.random.default_rng(42)
        for person_i, b in zip(idx_sorted[si[idx_sorted] >= 0], slot):
            if kind == "categorical":
                out[person_i] = labels[b]
            else:
                src = bin_vals[b]
                if len(src) == 0:  # empty source bin: use label representative
                    out[person_i] = labels[b]
                else:
                    k = min(cursor[b], len(src) - 1)
                    # sample evenly spaced quantiles of the bin values
                    q = (cursor[b] + 0.5) / target[b] if target[b] > 0 else 0.5
                    pos = min(int(q * (len(src) - 1) + 0), len(src) - 1)
                    v = float(src[pos])
                    # sim column stores RAW units; prep_variable log-transforms,
                    # so map log-space values back before writing (else the
                    # evaluator double-transforms)
                    if vcfg.get("log_transform") is True:
                        v = float(np.exp(v))
                    out[person_i] = v
                cursor[b] += 1
        return out

    for tag in args.beta_modes:
        sim_out = sim.copy()
        betas = {}
        for var, s in info.items():
            beta = beta_for(var, tag)
            betas[var] = round(beta, 4)
            sim_out[var] = reallocate(var, beta)
        sim_out.to_csv(os.path.join(d, f"sim_{tag}.csv"), index=False)
        shadow = os.path.join(d, f"run_{tag}")
        os.makedirs(shadow, exist_ok=True)
        sim_out.to_csv(os.path.join(shadow, "sim.csv"), index=False)
        if not os.path.exists(os.path.join(shadow, "real.csv")):
            shutil.copy(os.path.join(run_dir, "real.csv"), os.path.join(shadow, "real.csv"))
        with open(os.path.join(shadow, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"dataset": "cfps", "method": f"ada_{tag}",
                       "betas": betas, "source_run": run_dir, "ada_dir": d}, f, indent=2)
        print(f"[{tag}] betas: {betas}")
    print("done")


if __name__ == "__main__":
    main()
