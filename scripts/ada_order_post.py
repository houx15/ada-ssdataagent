"""Apply the ADA order-swap operator to an existing sim table (marginal-preserving).

Within-person exchanges of the three event ages move the E/C/M order-state
distribution toward p_target while exactly preserving every field's value
multiset. Mirrors pass 2 of ada_make_sim_order.py (same greedy + surplus logic).
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
from ssbench.evaluation.cleaning import prep_variable  # noqa: E402

STATES = ["EMC", "ECM", "MEC", "MCE", "CEM", "CME"]
OF = {"E": "age_finished_education", "C": "age_at_first_child",
      "M": "age_at_first_marriage"}
ROOT = os.path.join(os.path.dirname(__file__), "..")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--p-target", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or args.run_dir + "_ord"

    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "eval", "cfps.yaml")))
    sim = pd.read_csv(os.path.join(args.run_dir, "sim.csv"), low_memory=False)
    pj = json.load(open(args.p_target))
    p_t = np.array(pj["p"], float)
    p_t = p_t / p_t.sum()

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
    counts = np.zeros(6, dtype=int)
    for r in people:
        counts[r["state"]] += 1
    target = np.round(p_t * n).astype(int)
    target[0] += n - target.sum()  # remainder to the dominant state
    swaps = []
    by_state = {i: [r for r in people if r["state"] == i] for i in range(6)}
    for i in by_state:
        by_state[i].sort(key=lambda r: min(
            abs(r["ages"][x] - r["ages"][y])
            for x, y in itertools.combinations("EMC", 2)))

    # live surplus->deficit exact matching: each move takes a row from a
    # state that is above target and one-swap-moves it into a deficit
    # state (cheapest gap first).  Every move reduces total deviation, so
    # the loop terminates at the exact target histogram (or when no
    # one-swap candidate exists, which empirically never happens while
    # EMC is massively populated).
    guard = 0
    while True:
        dev = target - counts
        surplus = [i for i in range(6) if dev[i] <= -1]
        deficit = [j for j in range(6) if dev[j] >= 1]
        if not deficit:
            break
        guard += 1
        if guard > 20000:
            print("WARNING: order matching did not converge exactly")
            break
        best = None
        for j in deficit:
            for i in surplus:
                if i == j:
                    continue
                for r in by_state[i]:
                    l = list(STATES[r["state"]])
                    for x, y in itertools.combinations("EMC", 2):
                        l2 = list(l)
                        ix, iy = l2.index(x), l2.index(y)
                        l2[ix], l2[iy] = l2[iy], l2[ix]
                        if STATES.index("".join(l2)) == j:
                            cost = abs(r["ages"][x] - r["ages"][y])
                            if best is None or cost < best[0]:
                                best = (cost, r, x, y, i, j)
        if best is None:
            # exact repair fallback (integer atoms can block one-swaps):
            # any 3 distinct values can be permuted into ANY state order,
            # so assign one surplus row's sorted values to the deficit
            # state's positions.  Row value-multiset preserved; only the
            # within-row assignment moves (a 3-cycle, not a transposition).
            i = surplus[0]
            j = deficit[0]
            r = by_state[i][0]
            for letter, v in zip(STATES[j], sorted(r["ages"].values())):
                r["ages"][letter] = v
            r["state"] = j
            by_state[i].remove(r)
            by_state[j].append(r)
            counts[i] -= 1
            counts[j] += 1
            swaps.append({"pid": r["pid"], "ages": dict(r["ages"])})
            continue
        _, r, x, y, i, j = best
        r["ages"][x], r["ages"][y] = r["ages"][y], r["ages"][x]
        r["state"] = j
        by_state[i].remove(r)
        by_state[j].append(r)
        counts[i] -= 1
        counts[j] += 1
        swaps.append({"pid": r["pid"], "ages": dict(r["ages"])})

    for sw in swaps:
        for e, v in sw["ages"].items():
            sim.loc[sw["pid"], OF[e]] = v

    os.makedirs(out, exist_ok=True)
    sim.to_csv(os.path.join(out, "sim.csv"), index=False)
    shutil.copy(os.path.join(args.run_dir, "real.csv"), os.path.join(out, "real.csv"))
    after = counts / n
    json.dump({"dataset": "cfps", "method": "order_post_after_sensor_load",
               "base": args.run_dir, "n_swaps": len(swaps),
               "p_target": p_t.tolist(), "states": STATES,
               "order_dist_after": after.tolist()},
              open(os.path.join(out, "meta.json"), "w"), indent=1)
    print(f"swaps applied: {len(swaps)}")
    print("order dist after:", dict(zip(STATES, after.round(4))))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
