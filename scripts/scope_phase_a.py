"""SCOPE-Gen Phase A feasibility analysis (proposal §13 Phase A).

Tests the core hypothesis on existing runs: is the LLM channel distortion
low-order (K<=2 Chebyshev on graph/rank/quantile signals) and shared across
variables and datasets? Includes the identity-null falsification check.
"""

from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ssbench.evaluation.cleaning import clean_event_series, prep_variable
from ssbench.evaluation.fast_stats import (
    bincount2,
    cramers_v_from_counts,
    eta_squared_codes,
)
from ssbench.evaluation.t4_event_order import _order_label

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "analysis_out", "scope_phase_a")
RUNS = {
    "gss": os.path.join(ROOT, "runs", "gss", "direct", "20260817_163059_glm52"),
    "cfps": os.path.join(ROOT, "runs", "cfps", "direct", "20260817_164329_glm52"),
}

ORDINAL = {
    "gss": ["education", "income", "wealth", "health", "isolated", "lonely",
            "happy", "political_view", "satisfy_job", "work_hard"],
    "cfps": ["highest_education", "self_rated_health"],
}
NOMINAL = {
    "gss": ["marital_status", "laborforce", "occupation", "spouse_occupation",
            "depress", "gender_role_attitude", "trust"],
    "cfps": ["ever_divorced", "occupation_30_40"],
}
U_GRID = np.round(np.arange(0.05, 0.96, 0.05), 4)
ALPHA_PC = 0.5
EPS = 1e-3


def load_eval_cfg(ds: str) -> dict:
    with open(os.path.join(ROOT, "configs", "eval", f"{ds}.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_frames(ds: str):
    run = RUNS[ds]
    real = pd.read_csv(os.path.join(run, "real.csv"), low_memory=False)
    sim = pd.read_csv(os.path.join(run, "sim.csv"), low_memory=False)
    return real, sim


def path_laplacian_scaled(m: int) -> np.ndarray:
    L = np.diag(2 * np.ones(m)) - np.diag(np.ones(m - 1), 1) - np.diag(np.ones(m - 1), -1)
    L[0, 0] = L[-1, -1] = 1
    lam_max = np.linalg.eigvalsh(L)[-1]
    return 2.0 * L / lam_max - np.eye(m)


def cheb_features(y: np.ndarray, L_t: np.ndarray, K: int = 2) -> np.ndarray:
    cols = [y, L_t @ y]
    for _ in range(2, K + 1):
        cols.append(2.0 * (L_t @ cols[-1]) - cols[-2])
    return np.column_stack(cols)


def fit_ridge(blocks: list[tuple[np.ndarray, np.ndarray]], lam: float) -> np.ndarray:
    A = np.zeros((blocks[0][0].shape[1], blocks[0][0].shape[1]))
    b = np.zeros(blocks[0][0].shape[1])
    for Z, d in blocks:
        w = 1.0 / len(d)
        A += w * Z.T @ Z
        b += w * Z.T @ d
    return np.linalg.solve(A + lam * np.eye(A.shape[0]), b)


def tv(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * np.abs(p - q).sum()


def cat_block(p_counts: np.ndarray, q_counts: np.ndarray, order: list[str]):
    m = len(order)
    p = (p_counts + ALPHA_PC) / (p_counts.sum() + ALPHA_PC * m)
    q = (q_counts + ALPHA_PC) / (q_counts.sum() + ALPHA_PC * m)
    x = np.log(p) - np.log(p).mean()
    y = np.log(q) - np.log(q).mean()
    return p, q, x, y


def build_cat_blocks(ds: str, real: pd.DataFrame, sim: pd.DataFrame, cfg: dict):
    blocks = {}
    for kind, varlist in (("ordinal", ORDINAL[ds]), ("nominal", NOMINAL[ds])):
        for var in varlist:
            vcfg = dict(cfg["t1"]["variables"][var])
            cats = list(vcfg.get("allowed", []))
            r = prep_variable(real, var, vcfg).dropna()
            s = prep_variable(sim, var, vcfg).dropna()
            if r.empty or s.empty or not cats:
                continue
            cats_eff = [c for c in cats if (r == c).any() or (s == c).any()]
            pc = np.array([(r == c).sum() for c in cats_eff], dtype=float)
            qc = np.array([(s == c).sum() for c in cats_eff], dtype=float)
            p, q, x, y = cat_block(pc, qc, cats_eff)
            blocks[(kind, var)] = {
                "kind": kind, "var": var, "cats": cats_eff,
                "p": p, "q": q, "x": x, "y": y, "L": path_laplacian_scaled(len(cats_eff)),
            }
    return blocks


def equalize_cat(train_blocks: dict, train_keys: list, b: dict, lam: float,
                 K: int = 2, return_vec: bool = False):
    Zd = []
    for key in train_keys:
        tb = train_blocks[key]
        Zd.append((cheb_features(tb["y"], tb["L"], K), tb["x"] - tb["y"]))
    theta = fit_ridge(Zd, lam)
    d_hat = cheb_features(b["y"], b["L"], K) @ theta
    x_hat = b["y"] + d_hat
    p_hat = np.exp(x_hat - x_hat.max())
    p_hat /= p_hat.sum()
    if return_vec:
        return p_hat, d_hat
    return p_hat, float(np.abs(d_hat).mean()), float(np.abs(b["x"] - b["y"]).mean())


def equalize_nominal(train_blocks: dict, train_keys: list, b: dict, lam: float,
                     return_vec: bool = False):
    def feats(bb):
        m = len(bb["q"])
        order = np.argsort(-bb["q"])
        rank = np.empty(m)
        rank[order] = np.arange(m) / max(m - 1, 1)
        return np.column_stack([np.ones(m), rank, rank**2, np.log(bb["q"] + EPS)])

    Zd = [(feats(train_blocks[k]), train_blocks[k]["x"] - train_blocks[k]["y"]) for k in train_keys]
    gamma = fit_ridge(Zd, lam)
    d_hat = feats(b) @ gamma
    x_hat = b["y"] + d_hat
    p_hat = np.exp(x_hat - x_hat.max())
    p_hat /= p_hat.sum()
    if return_vec:
        return p_hat, d_hat
    return p_hat, float(np.abs(d_hat).mean()), float(np.abs(b["x"] - b["y"]).mean())


def quantiles(v: np.ndarray) -> np.ndarray:
    return np.quantile(v, U_GRID)


def build_num_blocks(ds: str, real: pd.DataFrame, sim: pd.DataFrame, cfg: dict):
    blocks = {}
    for var, vcfg in cfg["t1"]["variables"].items():
        if (vcfg.get("type") or "").lower() != "numeric":
            continue
        r = prep_variable(real, var, vcfg).dropna().to_numpy(float)
        s = prep_variable(sim, var, vcfg).dropna().to_numpy(float)
        if len(r) < 50 or len(s) < 50:
            continue
        med, iqr = np.median(s), np.subtract(*np.percentile(s, [75, 25]))
        if iqr <= 0:
            med, iqr = np.median(s), s.std()
        if iqr <= 0:
            continue
        yq = (quantiles(s) - med) / iqr
        xq = (quantiles(r) - med) / iqr
        lo, hi = vcfg.get("allowed", {}).get("min"), vcfg.get("allowed", {}).get("max")
        blocks[var] = {
            "var": var, "y": yq, "x": xq, "med": med, "iqr": iqr,
            "real": r, "sim": s, "lo": lo, "hi": hi,
            "L": path_laplacian_scaled(len(U_GRID)),
        }
    return blocks


def ecdf_sup_error(pred_q: np.ndarray, real_vals: np.ndarray) -> float:
    grid = np.sort(real_vals)
    cdf = np.interp(grid, pred_q, U_GRID, left=0.0, right=1.0)
    f_real = np.arange(1, len(grid) + 1) / len(grid)
    return float(np.abs(cdf - f_real).max())


def equalize_num(train_blocks: dict, train_keys: list, b: dict, lam: float,
                 K: int = 2, return_vec: bool = False):
    Zd = []
    for key in train_keys:
        tb = train_blocks[key]
        Zd.append((cheb_features(tb["y"], tb["L"], K), tb["x"] - tb["y"]))
    theta = fit_ridge(Zd, lam)
    d_hat = cheb_features(b["y"], b["L"], K) @ theta
    x_hat = b["y"] + d_hat
    x_hat = np.maximum.accumulate(x_hat)
    pred = b["med"] + b["iqr"] * x_hat
    if b["lo"] is not None:
        pred = np.clip(pred, b["lo"], None)
    if b["hi"] is not None:
        pred = np.clip(pred, None, b["hi"])
    pred = np.maximum.accumulate(pred)
    if return_vec:
        return pred, d_hat
    return pred, float(np.abs(d_hat).mean()), float(np.abs(b["x"] - b["y"]).mean())


def eval_cat_transfer(all_blocks: dict, train_ds: str, test_ds: str, lam: float) -> dict:
    rows = []
    for kind in ("ordinal", "nominal"):
        train_keys = [k for k in all_blocks[train_ds] if k[0] == kind]
        test_keys = [k for k in all_blocks[test_ds] if k[0] == kind]
        for tk in test_keys:
            b = all_blocks[test_ds][tk]
            fn = equalize_cat if kind == "ordinal" else equalize_nominal
            p_hat, dhat_mag, d_mag = fn(all_blocks[train_ds], train_keys, tk, lam)
            rows.append({
                "kind": kind, "var": tk[1], "train": train_ds, "test": test_ds,
                "tv_raw": tv(b["p"], b["q"]), "tv_eq": tv(b["p"], p_hat),
                "d_mag": d_mag, "dhat_mag": dhat_mag,
            })
    return rows


def eval_num_transfer(all_blocks: dict, train_ds: str, test_ds: str, lam: float) -> list:
    rows = []
    train_keys = list(all_blocks[train_ds])
    for tk in all_blocks[test_ds]:
        b = all_blocks[test_ds][tk]
        pred, dhat_mag, d_mag = equalize_num(all_blocks[train_ds], train_keys, tk, lam)
        ks_raw = ecdf_sup_error(quantiles(b["sim"]), b["real"])
        ks_eq = ecdf_sup_error(pred, b["real"])
        rows.append({
            "var": tk, "train": train_ds, "test": test_ds,
            "ks_raw": ks_raw, "ks_eq": ks_eq, "d_mag": d_mag, "dhat_mag": dhat_mag,
        })
    return rows


def identity_null(ds: str, real: pd.DataFrame, cfg: dict, lam: float) -> dict:
    rng = np.random.default_rng(7)
    idx = rng.permutation(len(real))
    h1, h2 = real.iloc[idx[: len(idx) // 2]], real.iloc[idx[len(idx) // 2:]]
    out = {"ds": ds, "cat": [], "num": []}
    for kind, varlist in (("ordinal", ORDINAL[ds]), ("nominal", NOMINAL[ds])):
        keys = []
        for var in varlist:
            vcfg = dict(cfg["t1"]["variables"][var])
            cats = list(vcfg.get("allowed", []))
            r = prep_variable(h1, var, vcfg).dropna()
            s = prep_variable(h2, var, vcfg).dropna()
            if r.empty or s.empty or not cats:
                continue
            cats_eff = [c for c in cats if (r == c).any() or (s == c).any()]
            pc = np.array([(r == c).sum() for c in cats_eff], dtype=float)
            qc = np.array([(s == c).sum() for c in cats_eff], dtype=float)
            p, q, x, y = cat_block(pc, qc, cats_eff)
            keys.append((kind, var, p, q, x, y, path_laplacian_scaled(len(cats_eff))))
        for i, (kind, var, p, q, x, y, L) in enumerate(keys):
            train = [(cheb_features(kk[5], kk[6]), kk[4] - kk[5])
                     for kk in keys if kk is not keys[i] and kk[0] == kind]
            if not train:
                continue
            Zb = cheb_features(y, L)
            theta = fit_ridge(train, lam)
            d_hat = Zb @ theta
            p_hat = np.exp(y + d_hat - (y + d_hat).max())
            p_hat /= p_hat.sum()
            out["cat"].append({
                "kind": kind, "var": var,
                "tv_raw": tv(p, q), "tv_eq": tv(p, p_hat),
                "dhat_mag": float(np.abs(d_hat).mean()), "d_mag": float(np.abs(x - y).mean()),
            })
    numblocks = []
    for var, vcfg in cfg["t1"]["variables"].items():
        if (vcfg.get("type") or "").lower() != "numeric":
            continue
        r = prep_variable(h1, var, vcfg).dropna().to_numpy(float)
        s = prep_variable(h2, var, vcfg).dropna().to_numpy(float)
        if len(r) < 30 or len(s) < 30:
            continue
        med, iqr = np.median(s), np.subtract(*np.percentile(s, [75, 25]))
        if iqr <= 0:
            continue
        numblocks.append((
            var, (quantiles(r) - med) / iqr, (quantiles(s) - med) / iqr,
            path_laplacian_scaled(len(U_GRID)),
        ))
    for i, (var, x, y, L) in enumerate(numblocks):
        train = [(cheb_features(kk[2], kk[3]), kk[1] - kk[2])
                 for j, kk in enumerate(numblocks) if j != i]
        if not train:
            continue
        theta = fit_ridge(train, lam)
        d_hat = cheb_features(y, L) @ theta
        out["num"].append({
            "var": var, "dhat_mag": float(np.abs(d_hat).mean()),
            "d_mag": float(np.abs(x - y).mean()),
        })
    return out


def assoc_blocks(ds: str, real: pd.DataFrame, sim: pd.DataFrame, cfg: dict):
    variables = cfg["t2"]["variables"]
    rows = []
    names = list(variables)
    for v1, v2 in itertools.combinations(names, 2):
        c1, c2 = variables[v1], variables[v2]
        if c1.get("input", False) and c2.get("input", False):
            continue
        t1t = (c1.get("type") or "categorical").lower()
        t2t = (c2.get("type") or "categorical").lower()
        atype = ("cat-cat" if t1t == t2t == "categorical"
                 else "num-num" if t1t == t2t == "numeric" else "num-cat")
        rec = {"ds": ds, "var1": v1, "var2": v2, "type": atype}

        def cleaned(df, v, c):
            return prep_variable(df, v, c).dropna()

        r1, s1 = cleaned(real, v1, c1), cleaned(sim, v1, c1)
        r2, s2 = cleaned(real, v2, c2), cleaned(sim, v2, c2)
        if atype == "cat-cat":
            def pair_table(df):
                d = pd.DataFrame({
                    "a": prep_variable(df, v1, c1), "b": prep_variable(df, v2, c2),
                }).dropna()
                return d["a"], d["b"]

            ra, rb = pair_table(real)
            sa, sb = pair_table(sim)
            cats1 = sorted(set(ra) | set(sa))
            cats2 = sorted(set(rb) | set(sb))
            i1 = {c: i for i, c in enumerate(cats1)}
            i2 = {c: i for i, c in enumerate(cats2)}
            ctab_r = bincount2(ra.map(i1).to_numpy(), rb.map(i2).to_numpy(), len(cats1), len(cats2))
            ctab_s = bincount2(sa.map(i1).to_numpy(), sb.map(i2).to_numpy(), len(cats1), len(cats2))
            if min(ctab_r.shape) < 2 or ctab_r.sum() < 50 or ctab_s.sum() < 50:
                continue

            def std_res(ct):
                p = ct / ct.sum()
                a, b_ = p.sum(1), p.sum(0)
                E = np.outer(a, b_)
                return (p - E) / np.sqrt(np.outer(a, b_) + 1e-12)

            Cp, Cq = std_res(ctab_r), std_res(ctab_s)
            Vp, _ = cramers_v_from_counts(ctab_r)
            Vq, _ = cramers_v_from_counts(ctab_s)
            cosv = float((Cp * Cq).sum() / (np.linalg.norm(Cp) * np.linalg.norm(Cq) + 1e-12))
            rec.update(V_P=Vp, V_Q=Vq, orient=cosv)
        elif atype == "num-num":
            def pair_corr(df):
                d = pd.DataFrame({
                    "a": prep_variable(df, v1, c1), "b": prep_variable(df, v2, c2),
                }).dropna()
                if len(d) < 50:
                    return np.nan
                return float(np.corrcoef(d["a"].astype(float), d["b"].astype(float))[0, 1])

            rp, rq = pair_corr(real), pair_corr(sim)
            if not (np.isfinite(rp) and np.isfinite(rq)):
                continue
            rec.update(V_P=rp, V_Q=rq, orient=np.sign(rp) * np.sign(rq))
        else:
            num_v, cat_v = (v1, v2) if t1t == "numeric" else (v2, v1)
            rn, cn = cleaned(real, num_v, variables[num_v]), cleaned(real, cat_v, variables[cat_v])
            sn, cn2 = cleaned(sim, num_v, variables[num_v]), cleaned(sim, cat_v, variables[cat_v])
            m = rn.index.intersection(cn.index)
            m2 = sn.index.intersection(cn2.index)
            if len(m) < 50 or len(m2) < 50:
                continue
            cats = sorted(set(cn[m]) | set(cn2[m2]))
            idx = {c: i for i, c in enumerate(cats)}
            ep, _ = eta_squared_codes(
                rn[m].to_numpy(float), cn[m].map(idx).to_numpy(), len(cats))
            eq, _ = eta_squared_codes(
                sn[m2].to_numpy(float), cn2[m2].map(idx).to_numpy(), len(cats))
            rec.update(V_P=ep, V_Q=eq, orient=np.nan)
        if np.isfinite(rec.get("V_P", np.nan)) and np.isfinite(rec.get("V_Q", np.nan)):
            rows.append(rec)
    return pd.DataFrame(rows)


def logit01(x: np.ndarray) -> np.ndarray:
    return np.log((x + EPS) / (1 - x + EPS))


def inv_logit01(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z)) * (1 - 2 * EPS) + EPS


def assoc_gain_eval(all_assoc: dict, train_ds: str, test_ds: str) -> list:
    rows = []
    tr, te = all_assoc[train_ds], all_assoc[test_ds]
    for atype, sub_tr in tr.groupby("type"):
        sub_te = te[te["type"] == atype]
        if len(sub_tr) < 5 or sub_te.empty:
            continue
        zq_p = logit01(np.abs(sub_tr["V_Q"].to_numpy()))
        zp_p = logit01(np.abs(sub_tr["V_P"].to_numpy()))
        X = np.column_stack([np.ones(len(zq_p)), zq_p])
        beta = np.linalg.lstsq(X, zp_p, rcond=None)[0]
        zq_te = logit01(np.abs(sub_te["V_Q"].to_numpy()))
        pred = inv_logit01(beta[0] + beta[1] * zq_te)
        rows.append(pd.DataFrame({
            "type": atype, "train": train_ds, "test": test_ds,
            "err_raw": np.abs(sub_te["V_Q"].to_numpy() - sub_te["V_P"].to_numpy()),
            "err_eq": np.abs(pred - sub_te["V_P"].to_numpy()),
            "V_P": sub_te["V_P"].to_numpy(), "V_Q": sub_te["V_Q"].to_numpy(),
        }))
    return rows


def t4_stats(real: pd.DataFrame, sim: pd.DataFrame, cfg: dict):
    events = cfg["t4"]["events"]
    combo = list(events)
    r = real.dropna(subset=combo, how="any").copy()
    s = sim.dropna(subset=combo, how="any").copy()
    for v in combo:
        r[v] = pd.to_numeric(clean_event_series(r[v], events[v]), errors="coerce")
        s[v] = pd.to_numeric(clean_event_series(s[v], events[v]), errors="coerce")
    r = r.dropna(subset=combo)
    s = s.dropna(subset=combo)
    r["order"] = r.apply(_order_label, axis=1, args=(combo,), sep="-")
    s["order"] = s.apply(_order_label, axis=1, args=(combo,), sep="-")
    pr = r["order"].value_counts(normalize=True)
    ps = s["order"].value_counts(normalize=True)
    states = sorted(set(pr.index) | set(ps.index))
    p = np.array([pr.get(x, 0.0) for x in states])
    q = np.array([ps.get(x, 0.0) for x in states])
    return {
        "tv": tv(p, q), "n_states_real": (p > 0).sum(), "n_states_sim": (q > 0).sum(),
        "entropy_real": float(-(p[p > 0] * np.log2(p[p > 0])).sum()),
        "entropy_sim": float(-(q[q > 0] * np.log2(q[q > 0])).sum()),
        "top_real": states[int(np.argmax(p))], "top_sim": states[int(np.argmax(q))],
        "p_top_real": float(p.max()), "p_top_sim": float(q.max()),
    }


def t3_stats(ds: str, real: pd.DataFrame, sim: pd.DataFrame, cfg: dict):
    import statsmodels.formula.api as smf

    preds = list(cfg["t3"]["predictors"])
    rows = []
    for y in cfg["t3"]["responses"]:
        ycfg = cfg["t3"]["responses"][y]

        def fit(df):
            d = pd.DataFrame({y: prep_variable(df, y, ycfg)})
            for p in preds:
                pcfg = cfg["t3"]["predictors"][p]
                s = prep_variable(df, p, pcfg)
                if (pcfg.get("type") or "").lower() == "categorical":
                    s = s.astype(str).mask(s.isin(["nan", "None"]), np.nan).astype("category")
                d[p] = s
            d = d.dropna()
            if len(d) < 60:
                return np.nan
            try:
                return float(smf.ols(f"{y} ~ {' + '.join(preds)}", data=d).fit().rsquared)
            except Exception:  # noqa: BLE001
                return np.nan

        rows.append({"ds": ds, "response": y, "R2_P": fit(real), "R2_Q": fit(sim)})
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    lam = 0.1
    cfgs = {ds: load_eval_cfg(ds) for ds in RUNS}
    frames = {ds: load_frames(ds) for ds in RUNS}

    cat_blocks = {ds: build_cat_blocks(ds, *frames[ds], cfgs[ds]) for ds in RUNS}
    num_blocks = {ds: build_num_blocks(ds, *frames[ds], cfgs[ds]) for ds in RUNS}

    report = {"lambda": lam, "n_blocks": {}}
    for ds in RUNS:
        report["n_blocks"][ds] = {
            "ordinal": sum(1 for k in cat_blocks[ds] if k[0] == "ordinal"),
            "nominal": sum(1 for k in cat_blocks[ds] if k[0] == "nominal"),
            "numeric": len(num_blocks[ds]),
        }

    cat_rows = []
    for tr, te in (("gss", "cfps"), ("cfps", "gss"), ("gss", "gss"), ("cfps", "cfps")):
        train_items = list(cat_blocks[tr].items())
        for tk, b in cat_blocks[te].items():
            train_keys = [k for k, _ in train_items if not (tr == te and k == tk)]
            if not train_keys:
                continue
            fn = equalize_cat if tk[0] == "ordinal" else equalize_nominal
            p_hat, d_hat_vec = fn(cat_blocks[tr], train_keys, b, lam, return_vec=True)
            d_vec = b["x"] - b["y"]
            alphas = np.array([0.3, 0.5, 0.7, 0.85, 1.0])
            def tv_flat(alpha):
                def apply(bb):
                    pf = bb["q"] ** alpha
                    return pf / pf.sum()
                errs = [tv(bb["p"], apply(bb)) for k, bb in train_items if k in train_keys]
                return np.mean(errs)
            best_alpha = float(alphas[np.argmin([tv_flat(a) for a in alphas])])
            p_flat = b["q"] ** best_alpha
            p_flat /= p_flat.sum()
            cat_rows.append({
                "kind": tk[0], "var": tk[1], "train": tr, "test": te,
                "tv_raw": tv(b["p"], b["q"]), "tv_eq": tv(b["p"], p_hat),
                "tv_flat": tv(b["p"], p_flat), "best_alpha": best_alpha,
                "improved": int(tv(b["p"], p_hat) < tv(b["p"], b["q"])),
                "beats_flat": int(tv(b["p"], p_hat) < tv(b["p"], p_flat)),
                "d_norm": float(np.linalg.norm(d_vec)),
                "resid_norm": float(np.linalg.norm(d_vec - d_hat_vec)),
            })
    cat_df = pd.DataFrame(cat_rows)
    cat_df.to_csv(os.path.join(OUT, "t1_cat_transfer.csv"), index=False)

    num_rows = []
    for tr, te in (("gss", "cfps"), ("cfps", "gss"), ("gss", "gss"), ("cfps", "cfps")):
        train_items = list(num_blocks[tr].items())
        for tk, b in num_blocks[te].items():
            train_keys = [k for k, _ in train_items if not (tr == te and k == tk)]
            pred, d_hat_vec = equalize_num(num_blocks[tr], train_keys, b, lam, return_vec=True)
            d_vec = b["x"] - b["y"]
            scales = np.array([1.25, 1.5, 2.0, 3.0, 4.0])
            def ks_scale(s):
                def apply(bb):
                    p = bb["med"] + bb["iqr"] * np.maximum.accumulate(s * bb["y"])
                    if bb["lo"] is not None:
                        p = np.clip(p, bb["lo"], None)
                    if bb["hi"] is not None:
                        p = np.clip(p, None, bb["hi"])
                    return np.maximum.accumulate(p)
                return np.mean([ecdf_sup_error(apply(bb), bb["real"]) for k, bb in train_items if k in train_keys])
            best_s = float(scales[np.argmin([ks_scale(s) for s in scales])])
            p_scaled = b["med"] + b["iqr"] * np.maximum.accumulate(best_s * b["y"])
            if b["lo"] is not None:
                p_scaled = np.clip(p_scaled, b["lo"], None)
            if b["hi"] is not None:
                p_scaled = np.clip(p_scaled, None, b["hi"])
            p_scaled = np.maximum.accumulate(p_scaled)
            num_rows.append({
                "var": tk, "train": tr, "test": te,
                "ks_raw": ecdf_sup_error(quantiles(b["sim"]), b["real"]),
                "ks_eq": ecdf_sup_error(pred, b["real"]),
                "ks_scale": ecdf_sup_error(p_scaled, b["real"]), "best_s": best_s,
                "improved": int(ecdf_sup_error(pred, b["real"]) < ecdf_sup_error(quantiles(b["sim"]), b["real"])),
                "beats_scale": int(ecdf_sup_error(pred, b["real"]) < ecdf_sup_error(p_scaled, b["real"])),
                "d_norm": float(np.linalg.norm(d_vec)),
                "resid_norm": float(np.linalg.norm(d_vec - d_hat_vec)),
            })
    num_df = pd.DataFrame(num_rows)
    num_df.to_csv(os.path.join(OUT, "t1_num_transfer.csv"), index=False)

    nulls = {ds: identity_null(ds, frames[ds][0], cfgs[ds], lam) for ds in RUNS}
    null_cat = pd.DataFrame([r for ds in RUNS for r in nulls[ds]["cat"]])
    null_num = pd.DataFrame([r for ds in RUNS for r in nulls[ds]["num"]])
    null_cat.to_csv(os.path.join(OUT, "null_cat.csv"), index=False)
    null_num.to_csv(os.path.join(OUT, "null_num.csv"), index=False)

    all_assoc = {ds: pd.DataFrame(assoc_blocks(ds, *frames[ds], cfgs[ds])) for ds in RUNS}
    for ds in RUNS:
        all_assoc[ds].to_csv(os.path.join(OUT, f"assoc_blocks_{ds}.csv"), index=False)
    assoc_rows = []
    for tr, te in (("gss", "cfps"), ("cfps", "gss")):
        assoc_rows.extend(assoc_gain_eval(all_assoc, tr, te))
    assoc_df = pd.concat(assoc_rows, ignore_index=True) if assoc_rows else pd.DataFrame()
    if not assoc_df.empty:
        assoc_df.to_csv(os.path.join(OUT, "assoc_transfer.csv"), index=False)

    t3_rows = []
    for ds in RUNS:
        t3_rows.extend(t3_stats(ds, *frames[ds], cfgs[ds]))
    t3_df = pd.DataFrame(t3_rows)
    t3_df.to_csv(os.path.join(OUT, "t3_r2.csv"), index=False)

    t4 = t4_stats(*frames["cfps"], cfgs["cfps"])

    print("\n" + "=" * 70)
    print("SCOPE Phase A — T1 categorical transfer (TV, lower better)")
    show = cat_df.groupby(["train", "test", "kind"]).agg(
        n=("tv_raw", "size"), tv_raw=("tv_raw", "mean"), tv_eq=("tv_eq", "mean"),
        tv_flat=("tv_flat", "mean"), win=("improved", "mean"), beats_flat=("beats_flat", "mean"),
        r2=("resid_norm", lambda r: 1 - (r**2).sum() / (cat_df.loc[r.index, "d_norm"]**2).sum()))
    print(show.round(3).to_string())
    print("\nT1 numeric transfer (KS sup error, lower better)")
    shown = num_df.groupby(["train", "test"]).agg(
        n=("ks_raw", "size"), ks_raw=("ks_raw", "mean"), ks_eq=("ks_eq", "mean"),
        ks_scale=("ks_scale", "mean"), win=("improved", "mean"), beats_scale=("beats_scale", "mean"),
        r2=("resid_norm", lambda r: 1 - (r**2).sum() / (num_df.loc[r.index, "d_norm"]**2).sum()))
    print(shown.round(3).to_string())

    print("\nIdentity null (should show ~no correction)")
    if not null_cat.empty:
        print(" cat: mean|d_hat| =", round(null_cat["dhat_mag"].mean(), 3),
              " vs null sampling noise |d| =", round(null_cat["d_mag"].mean(), 3),
              "| real-channel mean|d| =",
              round(pd.concat([
                  pd.Series([np.abs(b["x"] - b["y"]).mean() for b in cat_blocks[ds].values()])
                  for ds in RUNS]).mean(), 3),
              "| TV raw vs eq:", round(null_cat["tv_raw"].mean(), 3),
              "vs", round(null_cat["tv_eq"].mean(), 3))
    if not null_num.empty:
        print(" num: mean|d_hat| =", round(null_num["dhat_mag"].mean(), 3),
              " vs null sampling noise |d| =", round(null_num["d_mag"].mean(), 3),
              "| real-channel mean|d| =",
              round(pd.concat([
                  pd.Series([np.abs(b["x"] - b["y"]).mean() for b in num_blocks[ds].values()])
                  for ds in RUNS]).mean(), 3))

    print("\nAssociation blocks by dataset/type:")
    for ds in RUNS:
        print(" ", ds, all_assoc[ds].groupby("type").size().to_dict())

    if not assoc_df.empty:
        print("\nAssociation gain transfer (|V - V_P|, lower better)")
        ashow = assoc_df.groupby(["train", "test", "type"]).agg(
            n=("err_raw", "size"), err_raw=("err_raw", "mean"), err_eq=("err_eq", "mean"))
        ashow["win"] = (assoc_df.assign(better=assoc_df.err_eq < assoc_df.err_raw)
                        .groupby(["train", "test", "type"])["better"].mean())
        print(ashow.round(3).to_string())

    print("\nT3 R2 (P vs Q):")
    print(t3_df.round(3).to_string(index=False))

    print("\nT4 order distribution (CFPS):")
    for k, v in t4.items():
        print(f"  {k}: {v}")

    orient = {}
    for ds in RUNS:
        cc = all_assoc[ds][all_assoc[ds]["type"] == "cat-cat"]
        orient[ds] = {"n": len(cc), "orient_pos_rate": float((cc["orient"] > 0).mean()),
                      "orient_mean": float(cc["orient"].mean())}
    print("\nOrientation (cat-cat):", json.dumps(orient, indent=2))
    report.update({"t4": t4, "orient": orient})
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[phase_a] outputs -> {OUT}")


if __name__ == "__main__":
    main()
