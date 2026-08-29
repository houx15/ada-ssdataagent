"""Iman–Conover copula loading of T2 association targets onto a run (v1).

Input : a run dir with sim.csv/real.csv (e.g. run_order_fused7 — i.e. AFTER
all marginal loading, BEFORE the order swap).
Probe : runs/ada/t2_probe/full.jsonl (ada_t2_probe_full.py; pooling rule in
its docstring, fixed before any evaluation of the loaded output).

Algorithm (pre-registered):
 1. Ordinal score per T2 field (explicit maps; numeric = raw value,
    income in log1p — ranks are what matter).
 2. Target Spearman matrix R over the 23 t2 fields:
      diag 1; R[fixed, fixed] = empirical normal-score correlation of the
      four INPUT columns (inputs are given to the generator row-wise, so
      their joint structure is generator-side information, not held-out
      statistics); all other entries = probed rho (0 by default if a pair
      somehow has no probe). Symmetrise by averaging.
 3. Nearest positive-definite projection (eigenvalue clip at 1e-6).
 4. Draw Y ~ MVN(0, R) with n rows (fixed seed 20260819). Replace the four
    fixed columns of Y by the van der Waerden scores of the ACTUAL input
    columns (row-aligned). Reorder every free column so its ranks follow
    the corresponding Y column. Marginals of every free column are exactly
    preserved (pure permutation); input columns untouched.
 5. Write out/sim.csv (+ copy real.csv, meta.json with target matrix).

The order swap (ada_order_post.py) runs AFTER this step, as always.
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
from scipy.stats import norm, multivariate_normal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from ada_t2_probe_full import F as PROBE_FIELDS, INPUTS, LAM_EXT, G_HI, G_LO, GRID  # noqa

SEED = 20260819
ROOT = os.path.join(os.path.dirname(__file__), "..")

EDU_ORD = {"primary school or below": 0, "middle school": 1, "high school": 2,
           "college and above": 3}
HEALTH_ORD = {"very unhealthy": 0, "unhealthy": 1, "somewhat unhealthy": 2,
              "fairly healthy": 3, "very healthy": 4}
OCC_ORD = {  # ISCO listing order in cfps.yaml
    "Legislators, senior officials and managers": 0,
    "Professionals": 1,
    "Technicians and associate professionals": 2,
    "Clerks": 3,
    "Service workers and shop and market sales workers": 4,
    "Skilled agricultural and fishery workers": 5,
    "Craft and related trades workers": 6,
    "Plant and machine operators and assemblers": 7,
    "Elementary occupations": 8,
    "Armed forces": 9,
}


def ordinal_scores(df: pd.DataFrame, col: str):
    if col in ("highest_education", "mother_education", "father_education"):
        return df[col].map(EDU_ORD)
    if col == "self_rated_health":
        return df[col].map(HEALTH_ORD)
    if col == "occupation_30_40":
        return df[col].map(OCC_ORD)
    if col == "gender":
        return df[col].map({"Male": 1.0, "Female": 0.0})
    if col == "minzu":
        return df[col].map({"han": 1.0, "minority": 0.0})
    if col == "ever_divorced":
        return df[col].map({"never divorced": 0.0, "ever divorced": 1.0})
    v = pd.to_numeric(df[col], errors="coerce")
    if col == "mean_income_30_40":
        v = np.log1p(v.clip(lower=0))
    return v


def pooled_targets(jsonl: str) -> dict[tuple[str, str], float]:
    """Pool probe answers -> per unordered pair rho (rule fixed in probe docstring)."""
    import collections
    est = collections.defaultdict(lambda: collections.defaultdict(list))
    for line in open(jsonl):
        r = json.loads(line)
        if not r.get("parse_ok"):
            continue
        ch, content = r["combo"], r["content"]
        if r["design"] == "ratio":
            try:
                j = json.loads(content)
                vals = {}
                for a in j["answers"]:
                    rr = float(a.get("ratio", 1.0))
                    if not np.isfinite(rr) or rr <= 0:
                        rr = 1.0
                    vals[a.get("qid", "")] = float(np.clip(rr, 0.05, 50.0))
            except Exception:
                continue
            for i, (x, y) in enumerate(ch):
                lam = np.log(vals.get(f"q{i:02d}", 1.0))
                est[tuple(sorted((x, y)))]["ratio"].append(
                    float(np.interp(lam, LAM_EXT, GRID)))
        else:
            try:
                j = json.loads(content)
                vals = {}
                for a in j["answers"]:
                    av, bv = float(a.get("a_count", 50)), float(a.get("b_count", 50))
                    if not (0 <= av <= 100 and 0 <= bv <= 100):
                        av = bv = 50.0
                    vals[a.get("qid", "")] = (av / 100.0, bv / 100.0)
            except Exception:
                continue
            for i, (x, y) in enumerate(ch):
                a, b = vals.get(f"q{i:02d}", (0.5, 0.5))
                a = np.clip(a, 0.02, 0.98); b = np.clip(b, 0.02, 0.98)
                r_a = np.interp(a, G_HI, GRID)
                r_b = np.interp(b, G_LO[::-1], GRID[::-1])
                est[tuple(sorted((x, y)))]["count"].append(
                    float(np.tanh(np.mean(np.arctanh([r_a, r_b])))))

    out = {}
    for pair, dd in est.items():
        rho_r = (np.tanh(np.mean(np.arctanh(dd["ratio"])))
                 if dd["ratio"] else None)
        rho_c = (np.tanh(np.mean(np.arctanh(dd["count"])))
                 if dd["count"] else None)
        if rho_r is not None and rho_c is not None:
            if rho_r * rho_c < 0:          # sign conflict -> weaker evidence
                rho = rho_r if abs(rho_r) < abs(rho_c) else rho_c
            else:
                rho = float(np.tanh(np.mean(np.arctanh([rho_r, rho_c]))))
        else:
            rho = rho_r if rho_r is not None else rho_c
        out[pair] = float(rho) if rho is not None else 0.0
    return out


def van_der_waerden(x: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x)
    out = np.full_like(x, np.nan, dtype=float)
    ranks = pd.Series(x[ok]).rank().to_numpy()
    out[ok] = norm.ppf((ranks - 0.5) / ok.sum())
    return out


def nearest_psd(R: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    R = (R + R.T) / 2
    w, V = np.linalg.eigh(R)
    w = np.clip(w, eps, None)
    R2 = V @ np.diag(w) @ V.T
    d = np.sqrt(np.diag(R2))
    return R2 / np.outer(d, d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--probe", default="runs/ada/t2_probe/full.jsonl")
    ap.add_argument("--targets-json", default=None,
                    help="override pooled v1 targets with a fuse2-style JSON")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "eval", "cfps.yaml")))
    t2vars = list(cfg["t2"]["variables"])
    sim = pd.read_csv(os.path.join(args.run_dir, "sim.csv"), low_memory=False)

    if args.targets_json:
        targets = {}
        for t in json.load(open(args.targets_json)):
            a, b = t["pair"]
            targets[tuple(sorted((a, b)))] = float(t["rho"])
        print(f"targets from {args.targets_json}: {len(targets)}")
    else:
        targets = pooled_targets(args.probe)
        print(f"probed pairs: {len(targets)}")

    n = len(sim)
    idx = {v: i for i, v in enumerate(t2vars)}
    R = np.eye(len(t2vars))
    missing = []
    for i, a in enumerate(t2vars):
        for j in range(i + 1, len(t2vars)):
            b = t2vars[j]
            if a in INPUTS and b in INPUTS:
                continue  # filled from data below
            rho = targets.get(tuple(sorted((a, b))))
            if rho is None:
                missing.append((a, b)); rho = 0.0
            R[i, j] = R[j, i] = rho
    if missing:
        print(f"WARN missing probes (set 0): {missing}")

    # fixed-fixed block: empirical normal-score correlation of input columns
    fixed_cols = [v for v in t2vars if v in INPUTS]
    Zfixed = np.column_stack([van_der_waerden(
        ordinal_scores(sim, c).to_numpy(dtype=float)) for c in fixed_cols])
    Sff = np.corrcoef(Zfixed, rowvar=False)
    for i, a in enumerate(fixed_cols):
        for j, b in enumerate(fixed_cols):
            if j > i:
                R[idx[a], idx[b]] = R[idx[b], idx[a]] = Sff[i, j]

    R = nearest_psd(R)
    ev = np.linalg.eigvalsh(R)
    print(f"target R: max|rho|={np.abs(R[np.triu_indices_from(R,1)]).max():.3f} "
          f"eig=[{ev.min():.3f},{ev.max():.3f}]")

    def reorder(sim_df: pd.DataFrame, Rmat: np.ndarray) -> pd.DataFrame:
        """Conditional-Gaussian Iman-Conover.

        Free-column target scores are drawn from the MVN conditional on the
        ACTUAL van der Waerden scores of the fixed input columns:
          Y_v | X_f ~ N(B x_f, C),  B = R_vf S_ff^-1,  C = R_vv - B R_fv
        so cross-correlations with the real inputs and free-free correlations
        are both targeted in expectation. Free columns are then permuted to
        follow the ranks of Y_v (marginals exactly preserved).
        """
        rng = np.random.default_rng(SEED)
        Sff_i = np.linalg.inv(Sff)
        Rvf = Rmat[[idx[v] for v in t2vars if v not in INPUTS],
                   :][..., [idx[c] for c in fixed_cols]]  # (nf x 4)
        Rvv = Rmat[np.ix_([idx[v] for v in t2vars if v not in INPUTS],
                          [idx[v] for v in t2vars if v not in INPUTS])]
        B = Rvf @ Sff_i                                  # (nf x 4)
        C = Rvv - B @ Rvf.T
        C = nearest_psd(C, 1e-4)
        # Latin-hypercube (stratified) standard-normal sample for variance
        # reduction: free-free correlations land closer to C than an iid draw
        nf = C.shape[0]
        u = (np.argsort(np.argsort(
            rng.random((n, nf)), axis=0), axis=0) + rng.random((n, nf))) / n
        z = norm.ppf(np.clip(u, 1e-9, 1 - 1e-9))
        L = np.linalg.cholesky(C + 1e-9 * np.eye(nf))
        V = z @ L.T
        Yfree = Zfixed @ B.T + V                          # (n x nf)
        Y = np.zeros((n, len(t2vars)))
        free_idx = [idx[v] for v in t2vars if v not in INPUTS]
        for k, c in enumerate(fixed_cols):
            Y[:, idx[c]] = Zfixed[:, k]
        for k, ii in enumerate(free_idx):
            Y[:, ii] = Yfree[:, k]
        out = sim_df.copy()
        for v in t2vars:
            if v in INPUTS:
                continue
            raw = out[v].to_numpy(copy=True)
            s = ordinal_scores(out, v).to_numpy(dtype=float)
            ok = np.isfinite(s)
            tgt_rank = np.argsort(np.argsort(Y[ok, idx[v]]))
            order = np.argsort(s[ok], kind="stable")
            vals_sorted = raw[ok][order]
            new = raw.copy()
            new[~ok] = raw[~ok]
            new[ok] = vals_sorted[tgt_rank]
            out[v] = new
        return out

    def achieved(df: pd.DataFrame) -> np.ndarray:
        S = pd.DataFrame({v: van_der_waerden(
            ordinal_scores(df, v).to_numpy(dtype=float)) for v in t2vars})
        return S.corr().to_numpy()

    # two passes, always from the ORIGINAL table (fused9 scheme — the
    # 6-pass variant distorts strong pairs via PSD re-projection):
    # pass 1 measures the discretisation attenuation, pass 2 applies the
    # per-pair correction to a FRESH reorder; keep the lower-MAE pass.
    sim_a = reorder(sim, R)
    A = achieved(sim_a)
    iu = np.triu_indices_from(R, 1)
    m = np.isfinite(A[iu])
    mae1 = float(np.abs(R[iu][m] - A[iu][m]).mean())
    print(f"pass 1: MAE={mae1:.3f}")
    best, mae_best = sim_a, mae1
    if mae1 > 0.02:
        R2 = nearest_psd(np.clip(R + (R - np.nan_to_num(A)), -0.95, 0.95))
        sim_b = reorder(sim, R2)
        A2 = achieved(sim_b)
        m2 = np.isfinite(A2[iu])
        mae2 = float(np.abs(R[iu][m2] - A2[iu][m2]).mean())
        print(f"pass 2 (corrected): MAE={mae2:.3f}")
        if mae2 < mae_best:
            best, mae_best = sim_b, mae2
    sim2 = best

    # sanity: marginals preserved (as sorted multisets of stringified values)
    for v in t2vars:
        if v in INPUTS:
            continue
        a = sorted(pd.Series(sim[v].to_numpy(), dtype=object).astype(str))
        b = sorted(pd.Series(sim2[v].to_numpy(), dtype=object).astype(str))
        assert a == b, f"marginal changed for {v}!"

    n_perm = sum(int((sim2[v].to_numpy() != sim[v].to_numpy()).sum())
                 for v in t2vars if v not in INPUTS)
    print(f"cells moved: {n_perm}")
    if args.dry_run:
        # report achieved vs target correlations (pairwise NaN-safe)
        S = pd.DataFrame({v: van_der_waerden(
            ordinal_scores(sim2, v).to_numpy(dtype=float)) for v in t2vars})
        A = S.corr().to_numpy()
        tt = R[np.triu_indices_from(R, 1)]
        aa = A[np.triu_indices_from(A, 1)]
        m = np.isfinite(aa)
        print(f"corr(target, achieved) = {np.corrcoef(tt[m], aa[m])[0,1]:.3f}; "
              f"MAE={np.abs(tt[m]-aa[m]).mean():.3f}; pairs={m.sum()}")
        return

    os.makedirs(args.out, exist_ok=True)
    sim2.to_csv(os.path.join(args.out, "sim.csv"), index=False)
    shutil.copy(os.path.join(args.run_dir, "real.csv"),
                os.path.join(args.out, "real.csv"))
    meta_path = os.path.join(args.out, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        try:
            meta = json.load(open(meta_path))
        except Exception:
            meta = {}
    src_meta = os.path.join(args.run_dir, "meta.json")
    if os.path.exists(src_meta):
        try:
            base = json.load(open(src_meta))
            base.update({"t2_load": {"seed": SEED, "target_R": R.tolist(),
                                     "vars": t2vars, "probe": args.probe}})
            meta = base
        except Exception:
            pass
    if not meta:
        meta = {"t2_load": {"seed": SEED, "target_R": R.tolist(),
                            "vars": t2vars, "probe": args.probe}}
    json.dump(meta, open(meta_path, "w"), indent=1)
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
