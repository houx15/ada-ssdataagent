#!/usr/bin/env python3
"""β calibration study for the ADA CFPS run (zero extra LLM calls).

Questions:
 1. Do stratified β (per-kind mean/median of oracle β, LOFO) beat the global β?
 2. Does quantile edge-aggregation (per edge: majority sign × q90 of |g|) de-saturate
    the Arbiter's compressed output better than the mean?

Writes betas_calibrated.json for the best config (consumable by ada_make_sim.py).

Usage:
  uv run python scripts/ada_beta_calibration.py --dir runs/ada/cfps_20260817_222857
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ssbench.ada.signals import clr_with_pseudo, hodge, softmax  # noqa: E402
from ssbench.evaluation.cleaning import prep_variable  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUN = os.path.join(ROOT, "runs", "cfps", "direct", "20260817_164329_glm52")
BETAS_GRID = np.round(np.arange(-16.0, 16.001, 0.05), 2)


def tv(p, q):
    return 0.5 * np.abs(p - q).sum()


def load_fields(d):
    levels = json.load(open(os.path.join(d, "levels.json"), encoding="utf-8"))
    units = [json.loads(l) for l in open(os.path.join(d, "units.jsonl"), encoding="utf-8")]
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "eval", "cfps.yaml"), encoding="utf-8"))
    real = pd.read_csv(os.path.join(RUN, "real.csv"), low_memory=False)
    sim = pd.read_csv(os.path.join(RUN, "sim.csv"), low_memory=False)

    g_all = {}
    for u in units:
        for k, v in u["g"].items():
            var, jk = k.split("|")
            j, kk = map(int, jk.split(","))
            g_all.setdefault(var, {}).setdefault((j, kk), []).append(v)

    fields = {}
    for var, lm in levels.items():
        labels, kind, meta = lm["labels"], lm["kind"], lm.get("meta", {})
        m = len(labels)
        if var not in g_all:
            continue
        vcfg = dict(cfg["t1"]["variables"][var])

        def to_idx(values):
            out = []
            for v in values:
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

        si = to_idx(prep_variable(sim, var, vcfg).to_numpy())
        ri = to_idx(prep_variable(real, var, vcfg).to_numpy())
        if (si >= 0).sum() < 50 or (ri >= 0).sum() < 50:
            continue
        qc = np.bincount(si[si >= 0], minlength=m).astype(float)
        pc = np.bincount(ri[ri >= 0], minlength=m).astype(float)
        fields[var] = {"kind": kind, "m": m, "g": g_all[var],
                       "x_q": clr_with_pseudo(qc), "x_p": clr_with_pseudo(pc),
                       "p": pc / pc.sum(), "q": qc / qc.sum()}
    return fields


def phi_of(field, agg):
    gbar = {}
    for e, vals in field["g"].items():
        vals = np.asarray(vals, float)
        if agg == "mean":
            gbar[e] = float(vals.mean())
        else:
            s = 1.0 if vals.mean() >= 0 else -1.0
            gbar[e] = s * float(np.quantile(np.abs(vals), 0.9))
    phi, _ = hodge(gbar, ["x"] * field["m"])
    return phi


def oracle_beta(x_q, r, p):
    tvs = [tv(softmax(x_q + b * r), p) for b in BETAS_GRID]
    i = int(np.argmin(tvs))
    return float(BETAS_GRID[i]), float(tvs[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    fields = load_fields(args.dir)

    rows = []
    best = None
    for agg in ("mean", "q90"):
        per = {}
        for var, f in fields.items():
            phi = phi_of(f, agg)
            r = phi - f["x_q"]
            b_or, tv_or = oracle_beta(f["x_q"], r, f["p"])
            per[var] = {"r": r, "b_or": b_or, "tv_or": tv_or,
                        "tv_raw": tv(f["p"], f["q"]),
                        "c": float(np.linalg.norm(phi) / max(np.linalg.norm(f["x_p"] - f["x_q"]), 1e-9))}
        variants = ("global", "kind_mean", "kind_med")
        for var, f in fields.items():
            row = {"agg": agg, "var": var, "kind": f["kind"], "m": f["m"],
                   "c": round(per[var]["c"], 4), "tv_raw": round(per[var]["tv_raw"], 3),
                   "tv_oracle": round(per[var]["tv_or"], 3), "beta_oracle": round(per[var]["b_or"], 2)}
            others = [v for v in per if v != var]
            same_kind = [v for v in others if fields[v]["kind"] == f["kind"]]
            betas = {
                "global": float(np.mean([per[v]["b_or"] for v in others])),
                "kind_mean": float(np.mean([per[v]["b_or"] for v in same_kind])) if len(same_kind) >= 2 else None,
                "kind_med": float(np.median([per[v]["b_or"] for v in same_kind])) if len(same_kind) >= 2 else None,
            }
            for name in variants:
                b = betas[name] if betas[name] is not None else betas["global"]
                row[f"beta_{name}"] = round(b, 2)
                row[f"tv_{name}"] = round(tv(softmax(f["x_q"] + b * per[var]["r"]), f["p"]), 3)
            rows.append(row)
        means = {n: np.mean([r[f"tv_{n}"] for r in rows if r["agg"] == agg]) for n in variants}
        print(f"\n===== agg={agg}  mean TV by β rule: " +
              "  ".join(f"{n}={v:.3f}" for n, v in means.items()) +
              f"  oracle={np.mean([r['tv_oracle'] for r in rows if r['agg']==agg]):.3f}" +
              f"  raw={np.mean([r['tv_raw'] for r in rows if r['agg']==agg]):.3f}")
        cand = min(variants, key=lambda n: means[n])
        if best is None or means[cand] < best[1]:
            best = (f"{agg}+{cand}", means[cand], agg, cand)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    for agg in ("mean", "q90"):
        print(f"\n----- agg={agg} -----")
        print(df[df["agg"] == agg].drop(columns=["agg"]).to_string(index=False))

    # persist best config
    agg, rule = best[2], best[3]
    betas = {}
    for var, f in fields.items():
        per_var = [r for r in rows if r["var"] == var and r["agg"] == agg][0]
        betas[var] = per_var[f"beta_{rule}"]
    out = {"config": best[0], "mean_tv": round(best[1], 4), "agg": agg, "rule": rule, "betas": betas}
    with open(os.path.join(args.dir, "betas_calibrated.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[best] {best[0]}  mean TV {best[1]:.3f}  -> betas_calibrated.json")


if __name__ == "__main__":
    main()
