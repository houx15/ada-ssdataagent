#!/usr/bin/env python3
"""Apply the T4 order correction to an ADA run dir (no LLM calls).

Step 1: rank-preserving marginal reallocation per T1 field (softmax(phi)).
Step 2: order swap operator — greedy min-cost single-swap migration from the
        Actor's (degenerate) order states toward softmax(phi_order) targets;
        swapping = exchanging two event ages within a person.
Step 3: one more T1 reallocation pass to pin marginals back (swap moves values
        across field margins).

Writes run_order/ (sim.csv, real.csv, meta.json) for the official evaluator.

Usage:
  uv run python scripts/ada_make_sim_order.py --dir runs/ada/cfps_<stamp>
"""

from __future__ import annotations

import argparse
import itertools
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
STATES = ["".join(p) for p in itertools.permutations("EMC")]
OF = {"E": "age_finished_education", "M": "age_at_first_marriage",
      "C": "age_at_first_child"}
ORDER_FIELD = "__order__"


def load_edges(d):
    units = [json.loads(l) for l in open(os.path.join(d, "units.jsonl"),
                                         encoding="utf-8")]
    gall = {}
    for u in units:
        for k, v in u["g"].items():
            var, jk = k.split("|")
            j, kk = map(int, jk.split(","))
            gall.setdefault(var, {}).setdefault((j, kk), []).append(v)
    return gall


def to_idx_factory(lm, var, cfg):
    labels, kind, meta = lm["labels"], lm["kind"], lm.get("meta", {})
    vcfg = dict(cfg["t1"]["variables"][var])

    def f(s):
        out = []
        for v in prep_variable(s, var, vcfg).to_numpy():
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
    return f


def reallocate(sim, var, lm, p_hat, cfg, rng=None):
    labels, kind, meta = lm["labels"], lm["kind"], lm.get("meta", {})
    m = len(labels)
    f = to_idx_factory(lm, var, cfg)
    si = f(sim)
    n = (si >= 0).sum()
    target = np.floor(p_hat * n).astype(int)
    rem = n - target.sum()
    order = np.argsort(-(p_hat * n - target))
    for b in range(rem):
        target[order[b % m]] += 1
    vcfg = dict(cfg["t1"]["variables"][var])
    vals = prep_variable(sim, var, vcfg).to_numpy()
    bin_vals = {b: (np.sort(vals[si == b].astype(float)) if kind != "categorical"
                    else None) for b in range(m)}
    out = np.full(len(vals), np.nan, dtype=object)
    idx_sorted = np.argsort(np.where(si >= 0, si, 10 ** 9), kind="stable")
    slot = np.concatenate([[b] * target[b] for b in range(m)]).astype(int)
    cursor = {b: 0 for b in range(m)}
    for person_i, b in zip(idx_sorted[si[idx_sorted] >= 0], slot):
        if kind == "categorical":
            out[person_i] = labels[b]
        else:
            src = bin_vals[b]
            if len(src) == 0:
                v = labels[b]
            else:
                q = (cursor[b] + 0.5) / target[b] if target[b] > 0 else 0.5
                pos = min(int(q * (len(src) - 1)), len(src) - 1)
                v = float(src[pos])
            if rng is not None:
                v += rng.uniform(0.0, 1e-3)  # break exact ties across fields
            # prep_variable log-transforms at load; the sim column must store
            # RAW units, so map log-space values back before writing (writing
            # the log-space value directly double-transforms at evaluation)
            if vcfg.get("log_transform") is True and v is not None:
                v = float(np.exp(v))
            out[person_i] = v
        cursor[b] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tag", default="run_order")
    ap.add_argument("--p-target", default=None,
                    help="explicit target order distribution JSON {states, p}; "
                         "overrides softmax(phi_order) from units (e.g. ada_t4_fuse output)")
    ap.add_argument("--p-override-file", default=None,
                    help="JSON {field: [p_bin,...]} overriding softmax(phi) for specific "
                         "T1 fields in pass 1 (e.g. sensor-fused income target)")
    args = ap.parse_args()
    d = args.dir
    p_overrides = {}
    if args.p_override_file:
        p_overrides = json.load(open(args.p_override_file, encoding="utf-8"))

    levels = json.load(open(os.path.join(d, "levels.json"), encoding="utf-8"))
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "eval", "cfps.yaml"),
                              encoding="utf-8"))
    sim = pd.read_csv(os.path.join(RUN, "sim.csv"), low_memory=False).copy()
    gall = load_edges(d)

    # ---- pass 1: T1 marginal reallocation (pure phi) ----
    rng = np.random.default_rng(42)
    for var, lm in levels.items():
        if var == ORDER_FIELD or var not in gall:
            continue
        m = len(lm["labels"])
        if var in p_overrides:
            p_hat = np.array(p_overrides[var], float)
            p_hat = p_hat / p_hat.sum()
            print(f"p_override[{var}]: {p_hat.round(4).tolist()}")
        else:
            gbar = {k: float(np.mean(v)) for k, v in gall[var].items()}
            phi, _ = hodge(gbar, ["x"] * m)
            p_hat = softmax(phi)
        sim[var] = reallocate(sim, var, lm, p_hat, cfg, rng=rng)

    # ---- pass 2: order swap operator ----
    if args.p_target:
        pj = json.load(open(args.p_target, encoding="utf-8"))
        if list(pj["states"]) != STATES:
            raise SystemExit(f"p_target states mismatch: {pj['states']}")
        p_t = np.array(pj["p"], float)
        p_t = p_t / p_t.sum()
        print(f"p_target from {args.p_target}")
    else:
        gbar_o = {k: float(np.mean(v)) for k, v in gall[ORDER_FIELD].items()}
        phi_o, _ = hodge(gbar_o, STATES)
        p_t = softmax(phi_o)

    vals = {e: prep_variable(sim, f, dict(cfg["t1"]["variables"][f]))
            for e, f in OF.items()}
    people = []
    for pid in sim.index:
        try:
            ages = {e: float(vals[e].loc[pid]) for e in "EMC"}
        except Exception:  # noqa: BLE001
            continue
        if any(pd.isna(v) for v in ages.values()):
            continue
        a = sorted(ages.values())
        if any(abs(a[i] - a[i + 1]) < 1e-9 for i in range(2)):
            continue
        s = "".join(k for k, _ in sorted(ages.items(), key=lambda kv: kv[1]))
        people.append({"pid": pid, "state": STATES.index(s), "ages": ages})
    n = len(people)
    counts = np.zeros(6)
    for r in people:
        counts[r["state"]] += 1
    need = p_t * n - counts
    # greedy: repeatedly satisfy the most-underweight single-swap-reachable state
    swaps = []
    rng = np.random.default_rng(42)
    by_state = {i: [r for r in people if r["state"] == i] for i in range(6)}
    for i in by_state:
        by_state[i].sort(key=lambda r: min(
            abs(r["ages"][x] - r["ages"][y])
            for x, y in itertools.combinations("EMC", 2)))
    while True:
        deficit = {j: need[j] for j in range(6) if need[j] >= 1}
        if not deficit:
            break
        j = max(deficit, key=lambda j: need[j])
        # candidate sources: states i with a single swap to j, sorted by swap cost
        cands = []
        for i in range(6):
            if i == j or not by_state[i]:
                continue
            for r in by_state[i]:
                l = list(STATES[r["state"]])
                for x, y in itertools.combinations("EMC", 2):
                    l2 = list(l)
                    ix, iy = l2.index(x), l2.index(y)
                    l2[ix], l2[iy] = l2[iy], l2[ix]
                    if STATES.index("".join(l2)) == j:
                        cands.append((abs(r["ages"][x] - r["ages"][y]), r, x, y))
        if not cands:
            need[j] = 0
            continue
        cands.sort(key=lambda c: c[0])
        _, r, x, y = cands[0]
        r["ages"][x], r["ages"][y] = r["ages"][y], r["ages"][x]
        old_state = r["state"]
        r["state"] = j
        for i in range(6):
            if i != j and r in by_state[i]:
                by_state[i].remove(r)
                break
        need[j] -= 1
        swaps.append({"pid": r["pid"], "swap": [x, y]})

    # surplus correction: the deficit loop can overshoot a state (single swaps
    # to satisfy one deficit may transitively overfill another). Move excess
    # people from overfilled states back to underfilled ones via the cheapest
    # single swap that lands in an underfilled state.
    def _counts():
        c = np.zeros(6)
        for rr in people:
            c[rr["state"]] += 1
        return c
    for _round in range(200):
        c = _counts()
        dev = p_t * n - c
        surplus = {i: -dev[i] for i in range(6) if dev[i] <= -0.5}
        deficit = {j: dev[j] for j in range(6) if dev[j] >= 0.5}
        if not surplus or not deficit:
            break
        moved = False
        best = None
        for i in surplus:
            for rr in by_state[i]:
                l = list(STATES[rr["state"]])
                for x, y in itertools.combinations("EMC", 2):
                    l2 = list(l)
                    ix, iy = l2.index(x), l2.index(y)
                    l2[ix], l2[iy] = l2[iy], l2[ix]
                    j2 = STATES.index("".join(l2))
                    if j2 in deficit:
                        cost = abs(rr["ages"][x] - rr["ages"][y])
                        gain = min(surplus[i], deficit[j2])
                        key = (-(gain - 0.001 * cost), cost)
                        if best is None or key < best[0]:
                            best = (key, rr, x, y, i, j2)
        if best is None:
            break
        _, rr, x, y, i, j2 = best
        rr["ages"][x], rr["ages"][y] = rr["ages"][y], rr["ages"][x]
        rr["state"] = j2
        by_state[i].remove(rr)
        by_state[j2].append(rr)
        swaps.append({"pid": rr["pid"], "swap": [x, y]})
        moved = True
        if not moved:
            break

    sim2 = sim.set_index("profile_id" if "profile_id" in sim.columns else sim.index)
    for sw in swaps:
        pid = sw["pid"]
        x, y = sw["swap"]
        sim2.loc[pid, OF[x]], sim2.loc[pid, OF[y]] = sim2.loc[pid, OF[y]], sim2.loc[pid, OF[x]]

    # NOTE: no marginal re-pinning pass — each swap only exchanges two values
    # within a person, so every field's value multiset (hence marginal) is
    # exactly preserved by construction. Re-running per-field reallocation
    # here would destroy the order structure it just created (verified bug).

    out = os.path.join(d, args.tag)
    os.makedirs(out, exist_ok=True)
    sim2.to_csv(os.path.join(out, "sim.csv"), index=False)
    if not os.path.exists(os.path.join(out, "real.csv")):
        shutil.copy(os.path.join(RUN, "real.csv"), os.path.join(out, "real.csv"))
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"dataset": "cfps", "method": "ada_order",
                    "n_swaps": len(swaps), "p_target": p_t.tolist(),
                    "p_target_source": args.p_target or "softmax(hodge(pooled order edges))",
                    "states": STATES, "source_run": RUN, "ada_dir": d}, f, indent=1)
    print(f"swaps applied: {len(swaps)}")
    after = np.zeros(6)
    sim2r = sim2.reset_index()
    v2 = {e: prep_variable(sim2r, f, dict(cfg["t1"]["variables"][f])) for e, f in OF.items()}
    v2 = {e: s.set_axis(sim2r["profile_id"].values) for e, s in v2.items()}
    for pid in sim2r["profile_id"]:
        try:
            ages = {e: float(v2[e].loc[pid]) for e in "EMC"}
        except Exception:  # noqa: BLE001
            continue
        if any(pd.isna(x) for x in ages.values()):
            continue
        s = "".join(k for k, _ in sorted(ages.items(), key=lambda kv: kv[1]))
        after[STATES.index(s)] += 1
    print("order dist after:", dict(zip(STATES, (after / after.sum()).round(4))))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
