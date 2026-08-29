"""Fuse v2 within-stratum probes into marginal rank-correlation targets.

Pre-registered reconstruction (fixed before evaluation of any loaded run
using its output):
  1. Partial matrix P: unit diagonal; P[i,j] = pooled within-stratum rho
     (atanh-mean over strata x roles x reps; ext-band lookup per answer).
  2. PSD-clip P (eigenvalue floor 1e-6).
  3. Marginal correlation R = D (P^{-1}) D,  D = diag(1/sqrt(diag(P^{-1}))).
     (Exact identity for full partial correlations; here the conditioning
     set was only {highest_education, gender} — documented approximation.)
  4. Clip to [-0.95, 0.95]; fixed-input diagonal block left as data-derived
     (ada_t2_load re-fills it).
  5. Missing pairs (gender x highest_education was not probed in v2):
     filled from the v1 marginal probe.
Output: runs/ada/t2_probe/targets_v2.json — {"pair": [a, b], "rho": float}.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from ada_t2_probe_full import F, INPUTS, all_pairs, LAM_EXT, GRID  # noqa: E402
from ada_t2_load import pooled_targets, nearest_psd  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = "runs/ada/t2_probe/targets_v2.json"


def main() -> None:
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "eval", "cfps.yaml")))
    t2v = list(cfg["t2"]["variables"])
    idx = {v: i for i, v in enumerate(t2v)}

    est = defaultdict(list)
    for line in open("runs/ada/t2_probe/cond.jsonl"):
        r = json.loads(line)
        try:
            j = json.loads(r["content"])
            vals = {}
            for a in j["answers"]:
                rr = float(a.get("ratio", 1.0))
                if not np.isfinite(rr) or rr <= 0:
                    rr = 1.0
                vals[a.get("qid", "")] = float(np.clip(rr, 0.05, 50.0))
        except Exception:
            continue
        if len(vals) < max(1, len(r["combo"]) // 2):
            continue
        for i, (x, y, s) in enumerate(r["combo"]):
            lam = np.log(vals.get(f"q{i:02d}", 1.0))
            est[tuple(sorted((x, y)))].append(float(np.interp(lam, LAM_EXT, GRID)))

    P = np.eye(len(t2v))
    n_pair = 0
    for (a, b), v in est.items():
        rho = float(np.tanh(np.mean(np.arctanh(v))))
        P[idx[a], idx[b]] = P[idx[b], idx[a]] = rho
        n_pair += 1
    print(f"v2 partials: {n_pair} pairs")
    P = nearest_psd(P, 1e-6)
    Pi = np.linalg.inv(P)
    d = np.sqrt(np.diag(Pi))
    R = Pi / np.outer(d, d)
    R = np.clip(R, -0.95, 0.95)

    v1 = pooled_targets("runs/ada/t2_probe/full.jsonl")
    targets = []
    for i, a in enumerate(t2v):
        for j in range(i + 1, len(t2v)):
            b = t2v[j]
            if a in INPUTS and b in INPUTS:
                continue
            key = tuple(sorted((a, b)))
            if key in est:
                rho = float(R[idx[a], idx[b]])
            else:
                rho = v1.get(key, 0.0)
            targets.append({"pair": list(key), "rho": rho})
    json.dump(targets, open(OUT, "w"), indent=1)
    m = np.array([t["rho"] for t in targets])
    print(f"written {OUT}: n={len(targets)} rho in [{m.min():+.3f},{m.max():+.3f}], "
          f"|rho|>0.3: {(np.abs(m)>0.3).sum()}")


if __name__ == "__main__":
    main()
