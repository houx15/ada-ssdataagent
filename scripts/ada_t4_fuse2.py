#!/usr/bin/env python3
"""Zero-leakage T4 belief fusion v2 (pre-registered rule).

Sensor families (equal weight in a log-pool; availability per state varies):
  p_pw   pairwise arbiter softmax(hodge(pooled order edges))   [all states]
  p_ha   hierarchical 1000-alloc (probe2 8 reps + probe3 8 reps) [all states]
  p_seq  generative factorization P(first)*P(second|first)      [all states]
  p_pct  disambiguated pct probes (probe2)                      [ECM/MEC/MCE]

Zeroing rule: if the seq1 first-event 1000-alloc MEDIAN is 0 for event X,
all states starting with X are zeroed (model's own sub-1/1000 declaration),
mass renormalized. No other zeroing.

Usage:
  uv run python scripts/ada_t4_fuse2.py \
      --probe2 runs/ada/freq_probe2/probe.jsonl \
      --probe3 runs/ada/freq_probe3/probe.jsonl \
      --pairwise-dir runs/ada/cfps_round1+2_pooled \
      --out runs/ada/t4_fused2/p_target.json
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
sys.path.insert(0, os.path.dirname(__file__))

from ssbench.ada.signals import hodge, softmax  # noqa: E402
from ada_t4_fuse import parse_alloc, parse_pct, pairwise_p  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATES = ["".join(p) for p in itertools.permutations("EMC")]
ORDER_FIELD = "__order__"


def seq_joint(probe3: list[dict]) -> tuple[np.ndarray | None, dict]:
    """p = mean P(first) * mean P(second|first); also return raw first-event counts."""
    first_counts = defaultdict(list)   # event -> list of 1000-counts
    second_counts = defaultdict(list)  # (first, event) -> list of 1000-counts
    for r in probe3:
        if r["kind"] == "seq1":
            try:
                out = json.loads(r["content"])
                for e in "EMC":
                    v = float(out.get(f"{e}_first", np.nan))
                    if np.isfinite(v) and v >= 0:
                        first_counts[e].append(v)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        elif r["kind"] == "seq2":
            first = r["param"]
            try:
                out = json.loads(r["content"])
                for e in "EMC":
                    if e == first:
                        continue
                    v = float(out.get(f"{e}_second", np.nan))
                    if np.isfinite(v) and v >= 0:
                        second_counts[(first, e)].append(v)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    if not first_counts or len(first_counts) < 3:
        return None, {}
    pf = {e: np.mean(first_counts[e]) for e in "EMC"}
    tot = sum(pf.values())
    if tot <= 0:
        return None, {}
    pf = {e: v / tot for e, v in pf.items()}
    p = np.zeros(6)
    for first in "EMC":
        others = [e for e in "EMC" if e != first]
        cs = [np.mean(second_counts.get((first, e), [np.nan])) for e in others]
        if any(np.isnan(c) for c in cs):
            # no conditional data: fall back to equal split
            cs = [1.0, 1.0]
        s = sum(cs)
        ps = {e: c / s for e, c in zip(others, cs)}
        for second in others:
            third = [e for e in "EMC" if e not in (first, second)][0]
            state = first + second + third
            p[STATES.index(state)] = pf[first] * ps[second]
    p /= p.sum()
    return p, dict(first_counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe2", required=True)
    ap.add_argument("--probe3", required=True)
    ap.add_argument("--pairwise-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-meta", action="store_true",
                    help="also log-pool the meta-arbiter distribution (ablation off by default)")
    args = ap.parse_args()

    r2 = [json.loads(l) for l in open(args.probe2, encoding="utf-8")]
    r3 = [json.loads(l) for l in open(args.probe3, encoding="utf-8")]

    # p_ha: all parsable global hier-allocs from both probes
    allocs = []
    for r in r2 + r3:
        if r.get("kind") == "alloc" and r.get("col") is None:
            a = parse_alloc(r["content"])
            if a:
                allocs.append(a)
    if not allocs:
        raise SystemExit("no parsable allocations")
    p_ha = np.array([np.mean([a[s] for a in allocs]) for s in STATES])

    # p_pct per state
    pct = defaultdict(list)
    for r in r2:
        if r.get("kind") == "pct" and r.get("col") is None and r.get("target") in STATES:
            p = parse_pct(r["content"])
            if p is not None:
                pct[r["target"]].append(p)

    # p_seq
    p_seq, first_counts = seq_joint(r3)

    # p_pw
    p_pw, resid = pairwise_p(args.pairwise_dir)

    # meta (ablation)
    p_meta = None
    metas = []
    for r in r3:
        if r.get("kind") == "meta":
            try:
                out = json.loads(r["content"])
                v = [float(out.get(s, 0.0)) for s in STATES]
                if all(x >= 0 for x in v) and sum(v) > 0:
                    metas.append(np.array(v) / sum(v))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    if metas:
        p_meta = np.mean(metas, axis=0)

    print(f"alloc reps: {len(allocs)} | pct states: {dict((k, len(v)) for k, v in pct.items())} "
          f"| seq ok: {p_seq is not None} | meta reps: {len(metas)}")
    for name, p in [("p_pw", p_pw), ("p_ha", p_ha), ("p_seq", p_seq), ("p_meta", p_meta)]:
        if p is not None:
            print(f"{name:8s}", dict(zip(STATES, p.round(4))))

    # ---- pre-registered equal-weight log-pool over available families ----
    acc = np.zeros(6)
    cnt = np.zeros(6)
    for p in [p_pw, p_ha, p_seq]:
        if p is None:
            continue
        lp = np.log(np.clip(p, 1e-6, None))
        acc += lp
        cnt += 1
    if args.include_meta and p_meta is not None:
        acc += np.log(np.clip(p_meta, 1e-6, None))
        cnt += 1
    p_fused = np.exp(acc / np.maximum(cnt, 1))
    for s, vals in pct.items():
        i = STATES.index(s)
        p_fused[i] = np.sqrt(p_fused[i] * float(np.mean(vals)))  # equal blend
    p_fused /= p_fused.sum()

    # ---- zeroing by first-event median-0 declaration ----
    if first_counts:
        for e in "EMC":
            med = float(np.median(first_counts[e])) if first_counts.get(e) else None
            if med == 0.0:
                for i, s in enumerate(STATES):
                    if s[0] == e:
                        p_fused[i] = 0.0
        p_fused /= p_fused.sum()

    print("p_fused:", dict(zip(STATES, p_fused.round(4))))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"states": STATES, "p": p_fused.tolist(),
                   "sensors": {"pairwise": p_pw.tolist(), "hier_alloc": p_ha.tolist(),
                               "seq": None if p_seq is None else p_seq.tolist(),
                               "meta": None if p_meta is None else p_meta.tolist(),
                               "pct": {s: float(np.mean(v)) for s, v in pct.items()},
                               "n_alloc_reps": len(allocs), "n_meta_reps": len(metas)},
                   "pairwise_dir": args.pairwise_dir,
                   "probe2": args.probe2, "probe3": args.probe3}, f, indent=1)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
