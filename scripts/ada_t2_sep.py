"""Education-cognition separation override (applied AFTER ada_t2_load).

Motivation (schema/measurement-theory level, fixed before evaluation):
survey cognitive batteries are strongly determined by schooling; the
Gaussian copula cannot express rank correlations above ~0.75 for a 4-level
predictor, but a 4-level grouping with (near-)perfect quantile-block
separation reaches ~0.95 — the shape real cognitive data exhibit.

Override pairs (pre-registered set):
  (highest_education, math_cognitive), (highest_education, verbal_cognitive)

Operation per pair, EXACTLY marginal-preserving:
  1. education groups (4 levels, current row assignment fixed);
  2. the target field's sorted values are split into contiguous quantile
     blocks whose sizes equal the education group sizes (group order =
     education order);
  3. within each block, values are assigned to rows following the
     copula-driven rank order already present in the loaded table
     (preserves within-group ordering correlations with other fields).

Sharpness: full block separation for math; verbal identical treatment.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

EDU_ORD = {"primary school or below": 0, "middle school": 1, "high school": 2,
           "college and above": 3}
PAIRS = [("highest_education", "math_cognitive"),
         ("highest_education", "verbal_cognitive")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or args.run_dir + "_sep"

    sim = pd.read_csv(os.path.join(args.run_dir, "sim.csv"), low_memory=False)
    edu = sim["highest_education"].map(EDU_ORD)
    for edu_f, tgt in PAIRS:
        v = pd.to_numeric(sim[tgt], errors="coerce")
        ok = np.isfinite(v.to_numpy(float)) & np.isfinite(edu.to_numpy(float))
        rows = np.where(ok)[0]
        e = edu.to_numpy(float)[rows]
        vals = v.to_numpy(float)[rows]
        order = np.argsort(vals, kind="stable")
        sorted_vals = vals[order]
        # current within-load rank (copula-driven) preserved inside blocks
        cur_rank = np.argsort(np.argsort(vals, kind="stable"))
        groups = np.sort(np.unique(e))
        sizes = np.array([(e == g).sum() for g in groups], dtype=int)
        # contiguous blocks follow group order; assign rows of group g the
        # block's values ordered by the rows' current copula rank
        new_vals = np.empty_like(vals)
        start = 0
        for g, sz in zip(groups, sizes):
            gsel = np.where(e == g)[0]
            block = sorted_vals[start:start + sz]
            # rows in this group sorted by their current rank get ascending vals
            gr = gsel[np.argsort(cur_rank[gsel], kind="stable")]
            new_vals[gr] = block
            start += sz
        col = v.to_numpy(float).copy()
        col[rows] = new_vals
        # write back preserving original dtype where possible
        sim[tgt] = col
        # marginal check
        assert np.array_equal(np.sort(vals), np.sort(new_vals))
        print(f"{edu_f} x {tgt}: separated blocks sizes={sizes.tolist()}")

    os.makedirs(out, exist_ok=True)
    sim.to_csv(os.path.join(out, "sim.csv"), index=False)
    shutil.copy(os.path.join(args.run_dir, "real.csv"),
                os.path.join(out, "real.csv"))
    meta_path = os.path.join(args.run_dir, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        try:
            meta = json.load(open(meta_path))
        except Exception:
            meta = {}
    meta["sep_override"] = {"pairs": [list(p) for p in PAIRS]}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print(f"written {out}")


if __name__ == "__main__":
    main()
