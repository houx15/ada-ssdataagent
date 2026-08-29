"""Pool T1 probe sensors into per-field target CDFs (pre-registered rule).

p_alloc = log-pool( mean(asc reps), mean(desc reps) )        [band masses]
numeric: F = 0.5*F_alloc + 0.5*F_quant  (F_quant = piecewise-linear through
         median p10/p25/p50/p75/p90 anchors, extended to schema endpoints)
categorical: F = F_alloc
Integer-kind fields keep the mixed pmf on integer support (exact inverse).
Output: runs/ada/t1_probe/targets.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from ada_t1_probe import FIELDS, parse  # noqa: E402

OUT = "runs/ada/t1_probe/targets.json"


def logpool(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    p = np.exp((np.log(np.clip(a, 1e-6, None)) + np.log(np.clip(b, 1e-6, None))) / 2)
    return p / p.sum()


def main() -> None:
    acc: dict[tuple[str, str], list] = {}
    for line in open("runs/ada/t1_probe/probe.jsonl"):
        r = json.loads(line)
        v = parse(r["content"], r["sensor"])
        if v is None:
            continue
        acc.setdefault((r["field"], r["sensor"]), []).append(v)

    targets = {}
    for fld, f in FIELDS.items():
        nb = len(f["edges"]) - (1 if f["kind"] != "cat" else 0)
        ok_len = lambda v: len(v) == nb  # noqa: E731
        reps_a = [v for v in acc.get((fld, "alloc_asc"), []) if ok_len(v)]
        reps_d = [v for v in acc.get((fld, "alloc_desc"), []) if ok_len(v)]
        if not reps_a or not reps_d:
            print(f"!! {fld}: missing alloc reps ({len(reps_a)}/{len(reps_d)}) — skipped")
            continue
        p_asc = np.mean([np.array(v) / sum(v) for v in reps_a], axis=0)
        p_desc = np.mean([np.array(v) / sum(v) for v in reps_d], axis=0)
        if len(p_asc) != len(p_desc):
            print(f"!! {fld}: band count mismatch — skipped")
            continue
        p_alloc = logpool(p_asc, p_desc)

        t = {"kind": f["kind"], "schema": f["schema"], "edges": list(f["edges"]),
             "p_alloc": p_alloc.tolist(), "n_asc": len(reps_a), "n_desc": len(reps_d)}

        if f["kind"] == "cat":
            targets[fld] = t
            print(f"{fld:22s} cat  n={len(reps_a)}/{len(reps_d)}  p={p_alloc.round(3).tolist()}")
            continue

        reps_q = acc.get((fld, "quant"), [])
        if reps_q:
            q = np.median(np.array(reps_q), axis=0)
            q = np.maximum.accumulate(np.clip(q, f["schema"][0], f["schema"][1]))
            t["quant"] = q.tolist()
        targets[fld] = t
        qtxt = [f"{x:g}" for x in t.get("quant", [])]
        print(f"{fld:22s} {f['kind']:3s} n={len(reps_a)}/{len(reps_d)}/{len(reps_q)}  "
              f"p={p_alloc.round(3).tolist()}  q={qtxt}")

    json.dump(targets, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(f"-> {OUT} ({len(targets)} fields)")


if __name__ == "__main__":
    main()
