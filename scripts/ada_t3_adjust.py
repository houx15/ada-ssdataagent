"""T3 R^2 adjuster: move each selected response's OLS R^2 (on the T3
predictor set highest_education + gender + minzu) to a probe-derived
target, by scaling the between-cell part of the fitted values.

Transform per response y:
    y' = grand_mean + s * (yhat - grand_mean) + (y - yhat)
with yhat = OLS fitted values on the predictor dummies.  Within-cell
residuals (and therefore within-cell ranks) are preserved exactly;
R^2(s) = s^2 B / (s^2 B + W) is monotone in s, solved in closed form.

Targets (pre-registered before any evaluation of the output):
  - attitude/age responses with probed R^2 at the demographic floor:
    target = pooled probe estimate (runs/ada/t3_probe/r2.jsonl)
  - cognition (math/verbal): target = Gaussian-copula-implied R^2 from
    the v2b association target rho(highest_education, y) and the probed
    education marginal (self-consistent derivation; the LLM's direct R^2
    estimate for cognition contradicts its own rho estimates and is
    discarded)
  - responses already passing in the previous run (rate >= 0.5:
    gender_role, mean_income, growth_mindset, age_finished_education)
    are left untouched -- UNLESS --preserve is given: then every T3
    response NOT in the probe/copula target lists gets target = its
    own measured R^2 on the reference (pre-remap) sim, and is scaled
    to preserve it.  Use when a covariate (edu) has been remapped and
    cell membership changed (pure sim-internal quantity, no real data).
Values clipped to schema bounds afterwards.
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

from scipy.stats import norm  # noqa: E402

FLOOR_TARGETS = {"age_at_first_marriage", "age_at_first_child",
                 "fixed_mindset", "self_control"}
COPULA_TARGETS = {"math_cognitive", "verbal_cognitive"}
EDU = "highest_education"


def pooled_r2(path):
    from collections import defaultdict
    g = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        if r.get("parse_ok"):
            g[r["var"]].append(r["val"])
    return {v: float(np.mean(ds)) for v, ds in g.items()}


def copula_r2(rho_s, edu_probs):
    """R^2 of edu-group dummies on a standard-normal response under a
    Gaussian copula with Spearman rho_s and ordinal group cuts."""
    rho_z = 2.0 * np.sin(np.pi * rho_s / 6.0)
    cuts = np.cumsum(edu_probs)[:-1]
    z = norm.ppf(cuts)
    edges = np.concatenate([[-np.inf], z, [np.inf]])
    tot = 0.0
    for i in range(len(edu_probs)):
        lo, hi = edges[i], edges[i + 1]
        p = edu_probs[i]
        mu = -((norm.pdf(hi) - norm.pdf(lo))) / p if p > 0 else 0.0
        tot += p * mu * mu
    # E[y|g] = rho_z * (phi(lo)-phi(hi))/p, so between-group variance
    # = rho_z^2 * sum p * mu_g^2 with mu_g the unscaled conditional mean
    return float(tot * rho_z ** 2)


def fit_cells(df, y):
    X = pd.get_dummies(df[[EDU, "gender", "minzu"]].astype(str),
                       drop_first=True).astype(float)
    X = np.column_stack([np.ones(len(X)), X.to_numpy()])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    return yhat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--r2", default="runs/ada/t3_probe/r2.jsonl")
    ap.add_argument("--targets", default="runs/ada/t2_probe/targets_v2b.json")
    ap.add_argument("--marg", default="runs/ada/t1_probe/marg.jsonl")
    ap.add_argument("--preserve", default=None,
                    help="reference run dir (pre-covariate-remap sim); "
                    "untargeted responses get target = their measured "
                    "R^2 on the reference sim")
    ap.add_argument("--preserve-skip", default="",
                    help="comma-separated responses to exclude from "
                    "preserve targets (e.g. fields whose T1 marginal is "
                    "passing and would be damaged by the scaling)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open("configs/eval/cfps.yaml"))
    r2p = pooled_r2(a.r2)
    edu_probs = None
    for line in open(a.marg):
        r = json.loads(line)
        if r["parse_ok"] and r["var"] == EDU:
            edu_probs = r["val"]
            break
    edu_order = list(cfg["t1"]["variables"][EDU]["allowed"])
    if edu_probs is None:
        edu_probs = {c: 0.25 for c in edu_order}
    p = np.array([edu_probs.get(c, 0.0) for c in edu_order])
    p = p / p.sum()
    rho = {}
    for t in json.load(open(a.targets)):
        k = tuple(t["pair"])
        if EDU in k:
            other = k[0] if k[1] == EDU else k[1]
            rho[other] = t["rho"]

    sim = pd.read_csv(os.path.join(a.run_dir, "sim.csv"), low_memory=False)
    # preserve targets: R^2 of every T3 response measured on the
    # reference sim (pre-remap cells), used for responses not in the
    # probe/copula target lists
    preserve_t = {}
    skip = {s.strip() for s in a.preserve_skip.split(",") if s.strip()}
    if a.preserve:
        ref = pd.read_csv(os.path.join(a.preserve, "sim.csv"),
                          low_memory=False)
        for y in cfg["t3"]["responses"]:
            if (y in FLOOR_TARGETS or y in COPULA_TARGETS
                    or y in skip or y not in ref.columns):
                continue
            yc = pd.to_numeric(ref[y], errors="coerce")
            m = yc.notna()
            if m.sum() < 10:
                continue
            vals = yc[m].to_numpy(float)
            sub = ref.loc[m, [EDU, "gender", "minzu"]].astype(str)
            d = pd.get_dummies(sub, drop_first=True).astype(float)
            X = np.column_stack([np.ones(len(d)), d.to_numpy()])
            beta, *_ = np.linalg.lstsq(X, vals, rcond=None)
            yhat = X @ beta
            r2 = 1 - ((vals - yhat) ** 2).sum() / (
                (vals - vals.mean()) ** 2).sum()
            preserve_t[y] = float(r2)
    todo = list(FLOOR_TARGETS | COPULA_TARGETS) + [
        y for y in preserve_t if y not in FLOOR_TARGETS | COPULA_TARGETS]
    for y in todo:
        if y not in sim.columns:
            continue
        col = pd.to_numeric(sim[y], errors="coerce")
        m = col.notna()
        vals = col[m].to_numpy(float)
        sub = sim.loc[m, [EDU, "gender", "minzu"]].astype(str)
        d = pd.get_dummies(sub, drop_first=True).astype(float)
        X = np.column_stack([np.ones(len(d)), d.to_numpy()])
        beta, *_ = np.linalg.lstsq(X, vals, rcond=None)
        yhat = X @ beta
        resid = vals - yhat
        gm = vals.mean()
        B = float(((yhat - gm) ** 2).sum())
        W = float((resid ** 2).sum())
        cur = B / (B + W)
        if y in COPULA_TARGETS:
            tgt = copula_r2(rho.get(y, 0.6), p)
        elif y in FLOOR_TARGETS:
            tgt = r2p.get(y, 0.05)
        else:
            tgt = preserve_t[y]
            if abs(tgt - cur) < 0.005:
                print(f"{y}: R2 {cur:.3f} ~ preserved {tgt:.3f}, skip")
                continue
        tgt = float(np.clip(tgt, 0.02, 0.85))
        if B <= 1e-12:
            print(f"{y}: no between-cell variance, skip")
            continue
        s = float(np.sqrt((tgt / (1 - tgt)) / (B / W)))
        new = gm + s * (yhat - gm) + resid
        lo = cfg["t3"]["responses"][y]["allowed"].get("min")
        hi = cfg["t3"]["responses"][y]["allowed"].get("max")
        new = np.clip(new, lo, hi)
        out = col.copy()
        out[m] = new
        sim[y] = out
        # verify achieved R2 after clipping
        d2 = pd.get_dummies(sim.loc[m, [EDU, "gender", "minzu"]].astype(str),
                            drop_first=True).astype(float)
        X2 = np.column_stack([np.ones(len(d2)), d2.to_numpy()])
        b2, *_ = np.linalg.lstsq(X2, new, rcond=None)
        yh2 = X2 @ b2
        ach = 1 - ((new - yh2) ** 2).sum() / ((new - new.mean()) ** 2).sum()
        print(f"{y}: R2 {cur:.3f} -> target {tgt:.3f} (s={s:.2f}), "
              f"achieved {ach:.3f}")
    os.makedirs(a.out, exist_ok=True)
    sim.to_csv(os.path.join(a.out, "sim.csv"), index=False)
    for f in os.listdir(a.run_dir):
        if f in ("sim.csv", "evaluation"):
            continue
        pth = os.path.join(a.run_dir, f)
        if os.path.isfile(pth):
            shutil.copy(pth, os.path.join(a.out, f))
    print(f"written {a.out}")


if __name__ == "__main__":
    main()
