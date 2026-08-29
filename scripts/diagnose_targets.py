#!/usr/bin/env python3
"""Compare elicited T1/T2/T3/T4 targets with a loaded synthetic table."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from ada_t1_load import pool_marginals  # noqa: E402
from ada_t2_load import ordinal_scores  # noqa: E402
from ada_t3_adjust import fit_cells, pooled_r2  # noqa: E402

STATES = ["".join(p) for p in itertools.permutations("EMC")]


def add(rows, module, item, metric, target, achieved):
    error = None if target is None or achieved is None else float(achieved - target)
    rows.append({"module": module, "item": item, "metric": metric,
                 "target": target, "achieved": achieved, "error": error,
                 "abs_error": abs(error) if error is not None else None})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--t1", required=True)
    ap.add_argument("--t2", required=True)
    ap.add_argument("--t3", required=True)
    ap.add_argument("--t4", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    sim = pd.read_csv(os.path.join(args.run_dir, "sim.csv"), low_memory=False)
    cfg = yaml.safe_load(open("configs/eval/cfps.yaml"))
    rows = []

    cat, num, atom = pool_marginals(args.t1)
    for var, probs in cat.items():
        achieved = sim[var].value_counts(normalize=True, dropna=True).to_dict()
        tv = 0.5 * sum(abs(float(probs.get(k, 0)) - float(achieved.get(k, 0)))
                       for k in set(probs) | set(achieved))
        add(rows, "t1", var, "categorical_tv", 0.0, float(tv))
    for var, deciles in num.items():
        x = pd.to_numeric(sim[var], errors="coerce").dropna()
        if len(x):
            aq = x.quantile(np.arange(0.1, 1.0, 0.1)).to_numpy(float)
            add(rows, "t1", var, "decile_mae", 0.0,
                float(np.mean(np.abs(aq - np.asarray(deciles, float)))))
    for var, probs in atom.items():
        achieved = pd.to_numeric(sim[var], errors="coerce").value_counts(normalize=True).to_dict()
        tv = 0.5 * sum(abs(float(probs.get(k, 0)) - float(achieved.get(k, 0)))
                       for k in set(probs) | set(achieved))
        add(rows, "t1", var, "atom_tv", 0.0, float(tv))

    t2 = json.load(open(args.t2, encoding="utf-8"))
    t2_errors = []
    for target in t2:
        a, b = target["pair"]
        if a not in sim or b not in sim:
            continue
        sa, sb = ordinal_scores(sim, a), ordinal_scores(sim, b)
        achieved = sa.corr(sb, method="spearman")
        if pd.notna(achieved):
            add(rows, "t2", f"{a}|{b}", "spearman", float(target["rho"]), float(achieved))
            t2_errors.append(abs(float(achieved) - float(target["rho"])))
    if t2_errors:
        add(rows, "t2", "all_pairs", "mae", 0.0, float(np.mean(t2_errors)))

    r2_targets = pooled_r2(args.t3)
    for var, target in r2_targets.items():
        if var not in sim:
            continue
        y = pd.to_numeric(sim[var], errors="coerce")
        ok = y.notna()
        if ok.sum() < 10:
            continue
        vals = y[ok].to_numpy(float)
        yhat = fit_cells(sim.loc[ok], vals)
        denom = float(((vals - vals.mean()) ** 2).sum())
        achieved = 1 - float(((vals - yhat) ** 2).sum()) / denom if denom > 0 else None
        add(rows, "t3", var, "r2", float(target), achieved)

    target4 = json.load(open(args.t4, encoding="utf-8"))
    fields = {"E": "age_finished_education", "M": "age_at_first_marriage",
              "C": "age_at_first_child"}
    counts = {s: 0 for s in STATES}
    eligible = 0
    for _, row in sim.iterrows():
        ages = {k: pd.to_numeric(pd.Series([row[v]]), errors="coerce").iloc[0]
                for k, v in fields.items()}
        if any(pd.isna(v) for v in ages.values()) or len(set(ages.values())) < 3:
            continue
        state = "".join(k for k, _ in sorted(ages.items(), key=lambda kv: kv[1]))
        counts[state] += 1
        eligible += 1
    if eligible:
        achieved4 = np.array([counts[s] / eligible for s in target4["states"]])
        targetp = np.array(target4["p"], float)
        add(rows, "t4", "six_state", "total_variation", 0.0,
            float(0.5 * np.abs(achieved4 - targetp).sum()))
        for state, target, achieved in zip(target4["states"], targetp, achieved4):
            add(rows, "t4", state, "share", float(target), float(achieved))

    os.makedirs(args.out_dir, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(os.path.join(args.out_dir, "target_achieved.csv"), index=False)
    summary = frame.groupby(["module", "metric"], as_index=False)["abs_error"].mean()
    summary.to_csv(os.path.join(args.out_dir, "target_achieved_summary.csv"), index=False)
    print(f"target diagnostics -> {args.out_dir} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
