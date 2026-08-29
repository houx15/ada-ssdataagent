#!/usr/bin/env python3
"""Pool units.jsonl from multiple ADA collect runs into one analysis dir.

Adaptive multi-round ADA: round 2 measures the round-1-corrected population
(different Actor states, same personas, same levels grid), so edges that were
unmeasurable in round 1 (e.g. minority event orders adjacent to non-EMC
states) acquire observations. Pooling per edge before Hodge/softmax is valid
because an Arbiter edge response depends only on (persona, candidate pair),
never on which population generated the query.

Requirements checked per pair of dirs:
  - levels.json byte-identical (same node indexing -> poolable edges)
  - protocol.json agrees on arbiter_scale / grid / order_pairs

Units with empty g (network failures) are dropped. Duplicate persona_id across
rounds is expected and fine: they are independent measurements.

Usage:
  uv run python scripts/ada_pool_rounds.py \
      --dirs runs/ada/round1 runs/ada/round2 --out runs/ada/round1+2_pooled
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ORDER_FIELD = "__order__"


def md5_file(p: str) -> str:
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="two or more ADA collect dirs (each with units.jsonl + levels.json)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if len(args.dirs) < 2:
        raise SystemExit("need at least two dirs to pool")

    hashes = {md5_file(os.path.join(d, "levels.json")) for d in args.dirs}
    if len(hashes) != 1:
        raise SystemExit(f"levels.json differ across dirs — edges NOT poolable: {args.dirs}")

    prots = []
    for d in args.dirs:
        pp = os.path.join(d, "protocol.json")
        prots.append(json.load(open(pp, encoding="utf-8")) if os.path.exists(pp) else {})
    for k in ("arbiter_scale", "grid", "order_pairs"):
        if len({p.get(k) for p in prots}) != 1:
            raise SystemExit(f"protocol.{k} differs across dirs — refusing to pool")

    os.makedirs(args.out, exist_ok=True)
    n_units = 0
    edge_obs: Counter = Counter()
    per_round = []
    with open(os.path.join(args.out, "units.jsonl"), "w", encoding="utf-8") as fo:
        for d in args.dirs:
            n_keep = 0
            for line in open(os.path.join(d, "units.jsonl"), encoding="utf-8"):
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not rec.get("g"):
                    continue
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_keep += 1
                for k in rec["g"]:
                    edge_obs[k] += 1
            per_round.append({"dir": d, "units_kept": n_keep})
            n_units += n_keep

    shutil.copy(os.path.join(args.dirs[0], "levels.json"),
                os.path.join(args.out, "levels.json"))
    with open(os.path.join(args.out, "protocol.json"), "w", encoding="utf-8") as f:
        json.dump(prots[0] | {"pooled_from": args.dirs}, f)
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"method": "ada_pool", "rounds": per_round,
                   "n_units": n_units, "n_units_per_round": per_round}, f, indent=1)

    field_edges = Counter()
    order_edges = Counter()
    for k, n in edge_obs.items():
        var, _ = k.split("|")
        (order_edges if var == ORDER_FIELD else field_edges)[k] = n
    print(f"pooled {n_units} units from {len(args.dirs)} rounds -> {args.out}")
    print(f"field edges: {len(field_edges)} distinct, obs/edge "
          f"p10={np.percentile(list(field_edges.values()), 10):.0f} "
          f"p50={np.percentile(list(field_edges.values()), 50):.0f}")
    print(f"order edges: {len(order_edges)} distinct: "
          f"{ {k: v for k, v in sorted(order_edges.items())} }")


if __name__ == "__main__":
    main()
