#!/usr/bin/env python3
"""Analyze ADA order measurement (T4) from a collect run with --order-pairs.

Computes the Arbiter's order-state belief softmax(phi_order) over the 6
event orders and compares with the sim (Actor) and real order distributions.

Usage:
  uv run python scripts/ada_order_analyze.py --dir runs/ada/cfps_<stamp>
"""

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

from ssbench.ada.signals import hodge, softmax  # noqa: E402
from ssbench.evaluation.cleaning import prep_variable  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUN = os.path.join(ROOT, "runs", "cfps", "direct", "20260817_164329_glm52")
ORDER_FIELD = "__order__"
ORDER_FIELDS = {"E": "age_finished_education",
                "M": "age_at_first_marriage",
                "C": "age_at_first_child"}
STATES = ["".join(p) for p in itertools.permutations("EMC")]


def order_dist(df: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, int]:
    vals = {e: prep_variable(df, fld, dict(cfg["t1"]["variables"][fld]))
            for e, fld in ORDER_FIELDS.items()}
    counts = np.zeros(len(STATES))
    n_tie = 0
    for pid in df.index:
        if any(pd.isna(vals[e].loc[pid]) for e in "EMC"):
            continue
        ages = {e: float(vals[e].loc[pid]) for e in "EMC"}
        a = sorted(ages.values())
        if any(abs(a[i] - a[i + 1]) < 1e-9 for i in range(len(a) - 1)):
            n_tie += 1
            continue
        s = "".join(k for k, _ in sorted(ages.items(), key=lambda kv: kv[1]))
        counts[STATES.index(s)] += 1
    return counts, n_tie


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    d = args.dir

    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "eval", "cfps.yaml"),
                              encoding="utf-8"))
    real = pd.read_csv(os.path.join(RUN, "real.csv"), low_memory=False)
    sim = pd.read_csv(os.path.join(RUN, "sim.csv"), low_memory=False)
    real.index = real["profile_id"] if "profile_id" in real.columns else real.index
    sim.index = sim["profile_id"] if "profile_id" in sim.columns else sim.index

    units = [json.loads(l) for l in open(os.path.join(d, "units.jsonl"),
                                         encoding="utf-8")]
    gall = {}
    n_personas_with_order = 0
    for u in units:
        has = False
        for k, v in u["g"].items():
            var, jk = k.split("|")
            if var != ORDER_FIELD:
                continue
            j, kk = map(int, jk.split(","))
            gall.setdefault((j, kk), []).append(v)
            has = True
        if has:
            n_personas_with_order += 1
    if not gall:
        raise SystemExit("no __order__ edges found — was the run collected with --order-pairs?")
    gbar = {k: float(np.mean(v)) for k, v in gall.items()}
    phi, resid = hodge(gbar, STATES)
    p_ada = softmax(phi)

    pc, tie_r = order_dist(real, cfg)
    qc, tie_s = order_dist(sim, cfg)
    p_real, p_sim = pc / pc.sum(), qc / qc.sum()

    tv = lambda a, b: 0.5 * np.abs(a - b).sum()
    print(f"order-edge observations: {sum(len(v) for v in gall.values())} "
          f"over {len(gall)} edges; personas with order pairs: {n_personas_with_order}")
    print(f"ties skipped: real={tie_r} sim={tie_s}\n")
    print(f"{'state':6s} {'real':>7s} {'sim':>7s} {'ADA phi':>8s}")
    for i, s in enumerate(STATES):
        print(f"{s:6s} {p_real[i]:7.3f} {p_sim[i]:7.3f} {p_ada[i]:8.3f}")
    print(f"\nTV(ADA, real)   = {tv(p_ada, p_real):.4f}")
    print(f"TV(sim, real)   = {tv(p_sim, p_real):.4f}")
    print(f"TV(ADA, sim)    = {tv(p_ada, p_sim):.4f}")
    out = {"tv_ada_real": tv(p_ada, p_real), "tv_sim_real": tv(p_sim, p_real),
           "tv_ada_sim": tv(p_ada, p_sim),
           "p_real": p_real.tolist(), "p_sim": p_sim.tolist(),
           "p_ada": p_ada.tolist(), "states": STATES,
           "edge_counts": {f"{k[0]},{k[1]}": len(v) for k, v in gall.items()}}
    with open(os.path.join(d, "order_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"-> {os.path.join(d, 'order_analysis.json')}")


if __name__ == "__main__":
    main()
