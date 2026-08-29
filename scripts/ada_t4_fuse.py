#!/usr/bin/env python3
"""Zero-leakage T4 belief fusion.

Sensors (all model-belief only, no real data anywhere):
  1. pairwise arbiter  p_pw  = softmax(hodge(pooled order edges))   [sharp, compressed]
  2. hierarchical alloc p_ha (1000-person two-stage probe, global)  [de-compressed]
  3. disambiguated pct probes p_dp per state (ECM/MEC/MCE)          [direct frequency]

Fusion rule (pre-registered, no tuning):
  equal-weight logarithmic pool over available sensors:
      log p_fused ∝ (1/K) Σ_k log p_k
  Zeroing: a state is set to 0 iff the hierarchical-alloc MEDIAN count
  (across reps) is exactly 0 — i.e. the model itself declares it sub-1/1000.
  Remaining mass renormalized.

Usage:
  uv run python scripts/ada_t4_fuse.py --probe runs/ada/freq_probe2/probe.jsonl \
      --pairwise-dir runs/ada/cfps_round1+2_pooled --out runs/ada/t4_fused/p_target.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ssbench.ada.signals import hodge, softmax  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATES = ["".join(p) for p in itertools.permutations("EMC")]
ORDER_FIELD = "__order__"


def parse_alloc(text: str) -> dict[str, float] | None:
    try:
        out = json.loads(text)
        counts = {s: float(out.get(s, np.nan)) for s in STATES[:1] + STATES[1:]}
        counts["EMC"] = float(out.get("n_std", np.nan))
        if any(np.isnan(v) for v in counts.values()):
            return None
        if any(v < 0 for v in counts.values()):
            return None
        total = sum(counts.values())
        if total <= 0:
            return None
        return {k: v / total for k, v in counts.items()}
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def parse_pct(text: str) -> float | None:
    try:
        out = json.loads(text)
        v = float(out.get("pct", np.nan))
        if np.isnan(v) or not (0.0 <= v <= 100.0):
            return None
        return v / 100.0
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def pairwise_p(dir_: str) -> np.ndarray:
    units = [json.loads(l) for l in open(os.path.join(dir_, "units.jsonl"),
                                         encoding="utf-8")]
    oe = defaultdict(list)
    for u in units:
        for k, v in u["g"].items():
            var, jk = k.split("|")
            if var != ORDER_FIELD:
                continue
            j, kk = map(int, jk.split(","))
            oe[(j, kk)].append(v)
    g = {k: float(np.mean(v)) for k, v in oe.items()}
    # residual diagnostic (internal consistency, zero-leakage)
    phi, resid = hodge(g, STATES)
    return softmax(phi), resid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, help="freq_probe2 probe.jsonl")
    ap.add_argument("--pairwise-dir", required=True, help="ADA collect dir (units.jsonl)")
    ap.add_argument("--out", required=True, help="output p_target json")
    ap.add_argument("--no-zero", action="store_true", help="disable median-0 zeroing")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.probe, encoding="utf-8")]
    allocs_global, pct_by_state, alloc_counts_global = [], defaultdict(list), defaultdict(list)
    for r in recs:
        if r["col"] is not None:
            continue
        if r["kind"] == "alloc":
            a = parse_alloc(r["content"])
            if a:
                allocs_global.append(a)
                for s in STATES:
                    if a[s] > 0 or True:
                        alloc_counts_global[s].append(round(a[s] * 1000))
        elif r["kind"] == "pct":
            p = parse_pct(r["content"])
            if p is not None:
                pct_by_state[r["target"]].append(p)

    if not allocs_global:
        raise SystemExit("no parsable global allocations")
    p_ha = np.array([np.mean([a[s] for a in allocs_global]) for s in STATES])
    print(f"hier-alloc reps parsed: {len(allocs_global)}")
    print("p_ha:", dict(zip(STATES, p_ha.round(4))))
    for s in ("ECM", "MEC", "MCE"):
        if pct_by_state[s]:
            print(f"pct[{s}]: mean={np.mean(pct_by_state[s]):.4f} "
                  f"reps={len(pct_by_state[s])} vals={[round(v,3) for v in pct_by_state[s]]}")

    p_pw, resid = pairwise_p(args.pairwise_dir)
    print("p_pw:", dict(zip(STATES, p_pw.round(4))))
    resid_total = float(sum(resid.values())) if isinstance(resid, dict) else float(resid)
    print(f"pairwise hodge residual (total squared): {resid_total:.4f}")

    # ---- equal-weight log pool over available sensors ----
    logs = [np.log(np.clip(p_pw, 1e-6, None)), np.log(np.clip(p_ha, 1e-6, None))]
    # pct sensors only cover their own state; use them as likelihood boosts on
    # those states by treating each as a Dirac-free constraint: mix the pct
    # estimate into the state's pooled logit with equal weight
    p_fused = np.exp(sum(logs) / len(logs))
    for s, vals in pct_by_state.items():
        i = STATES.index(s)
        # replace the state's mass with the mean of pct and current fused,
        # then renormalize (conservative equal blend)
        p_fused[i] = np.sqrt(p_fused[i] * float(np.mean(vals)))
    p_fused /= p_fused.sum()

    # ---- median-0 zeroing ----
    if not args.no_zero:
        zeroed = [s for s in STATES
                  if s != "EMC"
                  and float(np.median(alloc_counts_global[s])) == 0.0]
        if zeroed:
            for s in zeroed:
                p_fused[STATES.index(s)] = 0.0
            p_fused /= p_fused.sum()
            print("zeroed (model's own sub-1/1000 declaration):", zeroed)

    print("p_fused:", dict(zip(STATES, p_fused.round(4))))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"states": STATES, "p": p_fused.tolist(),
                   "sensors": {"pairwise": p_pw.tolist(), "hier_alloc": p_ha.tolist(),
                               "pct": {s: float(np.mean(v)) for s, v in pct_by_state.items()},
                               "n_alloc_reps": len(allocs_global)},
                   "pairwise_dir": args.pairwise_dir, "probe": args.probe}, f, indent=1)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
