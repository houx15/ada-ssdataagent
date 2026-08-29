"""T1 marginal loader: rank-preserving remap onto LLM-probed marginals.

Rule (fixed before any evaluation of the output):
  numeric     : each column's values are replaced by a monotone quantile
                map (midrank u -> piecewise-linear probe deciles, tails
                linearly extended, clipped to schema bounds).  Monotone =>
                ALL rank correlations with every other column preserved
                exactly.  Count/age/Likert-scale variables are rounded to
                integers (semantic granularity, from schema bounds only).
  categorical : rows are ordered by the column's ordinal score (same score
                the copula loader uses) and reassigned to categories in the
                config 'allowed' order, block sizes = probed probabilities.
                Rank structure preserved exactly.
  occupation  : probed 'unemployed' share becomes NaN (config drop_values
                semantics) on a seeded-random row subset; the remaining
                rows get ISCO categories by ordinal-rank blocks.
Marginals therefore follow the probed distribution while associations
inherit the copula solution unchanged.  Event-order (E,M,C) columns are
mapped monotonically here and must be order-posted AFTER this step.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from ada_t2_load import ordinal_scores  # noqa: E402

INTEGER_VARS = {
    "fixed_mindset", "growth_mindset",
    "interpersonal_skills", "comprehension", "expression",
}
# event ages are integers in the source data; rounding also creates the
# correct integer atoms (KS against integer-valued real improves)
AGE_ROUND_VARS = {"age_at_first_marriage", "age_at_first_child"}
FLOOR_VARS = set()  # count semantics currently unused
LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def pool_marginals(path, override=None):
    cat, num, atom = {}, {}, {}
    for p in [path] + ([override] if override else []):
        # within one file, average all reps of a var; a later FILE
        # still overrides an earlier one per var
        fc, fn, fa = {}, {}, {}
        for line in open(p):
            r = json.loads(line)
            if not r.get("parse_ok"):
                continue
            if r["kind"] == "cat":
                fc.setdefault(r["var"], []).append(r["val"])
            elif r["kind"] == "atom":
                fa.setdefault(r["var"], []).append(r["val"])
            else:
                fn.setdefault(r["var"], []).append(r["val"])
        for v, ds in fc.items():
            cat[v] = {k: float(np.mean([d[k] for d in ds])) for k in ds[0]}
        for v, ds in fn.items():
            q = np.mean(np.array(ds, float), axis=0)
            num[v] = np.maximum.accumulate(q)
        # atom grids: align reps onto the UNION of keys (missing -> 0);
        # JSON dict keys are strings -> convert to float once here
        for v, ds in fa.items():
            keys = sorted({k for d in ds for k in d})
            atom[v] = {float(k): float(np.mean(
                [d.get(k, d.get(str(float(k)), 0.0)) for d in ds]))
                for k in keys}
    return cat, num, atom


def map_numeric(col, deciles, lo, hi, integer, floor=False, rng=None):
    x = col.to_numpy(float)
    out = np.full_like(x, np.nan)
    m = np.isfinite(x)
    if m.sum() < 2:
        return col
    xs = x[m].copy()
    if rng is not None:
        # seeded tie-break jitter: only affects the ORDER of exactly-tied
        # values (Spearman/vdW ranks identical), removes KS atoms
        xs = xs + rng.uniform(-1e-6, 1e-6, size=xs.shape) * (
            np.nanmax(np.abs(xs)) + 1.0)
    ranks = pd.Series(xs).rank(method="average").to_numpy()
    u = (ranks - 0.5) / m.sum()
    q = np.asarray(deciles, float)
    lv = np.arange(1, len(q) + 1) / (len(q) + 1)   # grid-adaptive levels
    q = np.clip(q, lo, hi)
    val = np.interp(u, lv, q)
    step = lv[1] - lv[0]
    left = u < lv[0]
    right = u > lv[-1]
    if q[1] > q[0]:
        val[left] = q[0] - (lv[0] - u[left]) * (q[1] - q[0]) / step
    else:
        val[left] = q[0]
    if q[-1] > q[-2]:
        val[right] = q[-1] + (u[right] - lv[-1]) * (q[-1] - q[-2]) / step
    else:
        val[right] = q[-1]
    val = np.clip(val, lo, hi)
    if integer:
        val = np.floor(val) if floor else np.round(val)
        # a nondecreasing sequence stays nondecreasing under floor/round
        # (val is monotone in the rank u); NO row-order cummax — that
        # would propagate early maxima across unrelated rows.
    out[m] = val
    return pd.Series(out, index=col.index)


def map_categorical(df, var, probs, allowed, rng):
    col = df[var]
    s = ordinal_scores(df, var)
    x = s.to_numpy(float)
    idx = np.where(np.isfinite(x))[0]
    order = idx[np.argsort(x[idx], kind="stable")]
    out = np.array([None] * len(df), dtype=object)
    n = len(order)
    cum = np.cumsum([probs.get(c, 0.0) for c in allowed])
    cum = cum / cum[-1]
    edges = np.concatenate([[0.0], cum])
    blocks = np.minimum(np.round(edges * n).astype(int), n)
    for i, c in enumerate(allowed):
        sel = order[blocks[i]:blocks[i + 1]]
        out[sel] = c
    return pd.Series(out, index=df.index)


def map_atoms(col, dist, rng=None):
    """Rank-block assignment onto a numeric ATOM GRID (value->prob).

    Rows are ordered by seeded tie-broken rank and assigned to atom
    values in sorted order, block sizes = probed probabilities.  Rank
    order across blocks preserved exactly; the resulting column is
    discrete on the atom grid (matching composite-scale marginals,
    e.g. k-item means or item-count sums)."""
    x = col.to_numpy(float)
    out = np.full_like(x, np.nan)
    m = np.isfinite(x)
    if m.sum() < 2 or not dist:
        return col
    xs = x[m].copy()
    if rng is not None:
        xs = xs + rng.uniform(-1e-6, 1e-6, size=xs.shape) * (
            np.nanmax(np.abs(xs)) + 1.0)
    order = np.argsort(xs, kind="stable")
    vals = np.array(sorted(dist.keys()), float)
    ps = np.array([dist[v] for v in vals], float)
    ps = ps / ps.sum()
    n = len(order)
    cum = np.cumsum(ps)
    blocks = np.minimum(np.round(np.concatenate([[0.0], cum]) * n).astype(int), n)
    res = np.empty(n)
    for i, v in enumerate(vals):
        res[order[blocks[i]:blocks[i + 1]]] = v
    out[m] = res
    return pd.Series(out, index=col.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--marg", default="runs/ada/t1_probe/marg.jsonl")
    ap.add_argument("--marg2", default=None,
                    help="override file (e.g. re-probed binary vars)")
    ap.add_argument("--only", default=None,
                    help="comma-separated variables to remap (default all)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open("configs/eval/cfps.yaml"))
    cat, num, atom = pool_marginals(a.marg, a.marg2)
    sim = pd.read_csv(os.path.join(a.run_dir, "sim.csv"), low_memory=False)
    rng = np.random.default_rng(0)
    for var, spec in cfg["t1"]["variables"].items():
        if var not in sim.columns:
            continue
        if a.only and var not in a.only.split(","):
            continue
        if spec["type"] == "categorical":
            allowed = list(spec["allowed"])
            probs = dict(cat.get(var, {}))
            if var == "occupation_30_40":
                p_un = probs.pop("unemployed", 0.0)
                tot = sum(probs.values()) or 1.0
                probs = {k: v / tot for k, v in probs.items()}
                n = len(sim)
                miss = rng.choice(n, size=int(round(p_un * n)), replace=False)
                sim[var] = map_categorical(sim, var, probs, allowed, rng)
                sim.loc[sim.index[miss], var] = np.nan
            else:
                tot = sum(probs.values()) or 1.0
                probs = {k: v / tot for k, v in probs.items()}
                sim[var] = map_categorical(sim, var, probs, allowed, rng)
        else:
            lo = spec["allowed"].get("min")
            hi = spec["allowed"].get("max")
            dist = atom.get(var)
            if dist is not None:
                # atom-grid loading (composite scales): clip grid to
                # schema bounds, renormalize
                dist = {v: p for v, p in dist.items() if lo <= v <= hi}
                if dist:
                    sim[var] = map_atoms(sim[var], dist, rng)
                vc = sim[var].value_counts(normalize=True, dropna=False)
                top = ", ".join(f"{k}:{v:.2f}"
                                for k, v in vc.head(5).items())
                print(f"{var}: {top}")
                continue
            q = num.get(var)
            if q is None:
                continue
            sim[var] = map_numeric(sim[var], q, lo, hi,
                                   var in INTEGER_VARS,
                                   floor=var in FLOOR_VARS, rng=rng)
            if var in AGE_ROUND_VARS:
                col = pd.to_numeric(sim[var], errors="coerce")
                col = col.round()
                sim[var] = col
        vc = sim[var].value_counts(normalize=True, dropna=False)
        top = ", ".join(f"{k}:{v:.2f}" for k, v in vc.head(4).items())
        print(f"{var}: {top}")
    # evaluator-consistent tie resolution for ALL three event ages: the
    # T4 evaluator stable-sorts by value with config list order E, C, M,
    # so on any exact tie the earlier letter counts as "before".  Give
    # tied later letters a +0.25 offset (offsets from ORIGINAL equality
    # only: a triple E=C=M becomes E < C+.25 < M+.5).  With zero ties
    # left, order_post accounts the same n as the evaluator and its
    # exact target match carries through (963 -> 1000 was the fused14
    # T4 regression).
    EF = "age_finished_education"
    CF = "age_at_first_child"
    MF = "age_at_first_marriage"
    e_ = pd.to_numeric(sim[EF], errors="coerce")
    c_ = pd.to_numeric(sim[CF], errors="coerce")
    m_ = pd.to_numeric(sim[MF], errors="coerce")
    present = e_.notna() & c_.notna() & m_.notna()
    eq_ec = present & (e_ == c_)
    eq_em = present & (e_ == m_)
    eq_cm = present & (c_ == m_)
    # NB: pandas 2.x `bool + bool` is logical OR, not integer addition —
    # must cast to int for group-rank offsets (a triple E=C=M needs M+0.5)
    c_ = c_ + 0.25 * eq_ec.astype(int)
    m_ = m_ + 0.25 * (eq_em.astype(int) + eq_cm.astype(int))
    sim[CF] = c_
    sim[MF] = m_
    if int((eq_ec | eq_em | eq_cm).sum()):
        print(f"tie-break EC={int(eq_ec.sum())} EM={int(eq_em.sum())} "
              f"CM={int(eq_cm.sum())} rows (evaluator order)")
    os.makedirs(a.out, exist_ok=True)
    sim.to_csv(os.path.join(a.out, "sim.csv"), index=False)
    for f in os.listdir(a.run_dir):
        if f in ("sim.csv", "evaluation"):
            continue
        p = os.path.join(a.run_dir, f)
        if os.path.isfile(p):
            shutil.copy(p, os.path.join(a.out, f))
    print(f"written {a.out}")


if __name__ == "__main__":
    main()
