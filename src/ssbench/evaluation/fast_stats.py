"""Fast numpy implementations of the T1/T2/T5 test statistics.

Semantics are identical to the original SSDataBench tests (same estimators,
same delta-method variances, same degeneracy guards) — only the computation
is vectorized: analytic gradients instead of autograd, bincount contingency
tables instead of pandas crosstab, closed-form eta-squared instead of OLS.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

_EPS = 1e-20


def bincount2(x: np.ndarray, y: np.ndarray, nr: int, nc: int) -> np.ndarray:
    """2-D contingency table from integer codes via flat bincount."""
    flat = np.asarray(x, dtype=np.int64) * nc + np.asarray(y, dtype=np.int64)
    return np.bincount(flat, minlength=nr * nc).reshape(nr, nc)


def cramers_v_from_counts(ct: np.ndarray) -> tuple[float, float]:
    """Cramér's V and its multinomial delta-method variance (analytic gradient).

    Mirrors SSDataBench's ``_cramers_v_and_var_autograd``: phi2 uses the
    ``+1e-20`` epsilon, V = sqrt(phi2 / (q-1)) with q = min(r, c), and
    V < 1e-6 short-circuits to (0, 0).
    """
    cm = np.asarray(ct, dtype=float)
    n = cm.sum()
    r, c = cm.shape
    q = min(r, c)
    if n == 0 or q <= 1:
        return np.nan, np.nan

    p = cm / n
    a = p.sum(axis=1, keepdims=True)   # row sums
    b = p.sum(axis=0, keepdims=True)   # col sums
    e = a @ b
    phi2 = float(((p - e) ** 2 / (e + _EPS)).sum())
    V = float(np.sqrt(phi2 / (q - 1)))
    if V < 1e-6:
        return 0.0, 0.0

    # Analytic gradient of T = sum p^2 / (a_i b_j) w.r.t. p (phi2 = T - 1):
    # dT/dp_kl = 2 p_kl/(a_k b_l) - (1/a_k^2) sum_j p_kj^2/b_j - (1/b_l^2) sum_i p_il^2/a_i
    term_r = (p * p / (b + _EPS)).sum(axis=1, keepdims=True) / (a**2 + _EPS)  # (1/a_k^2) Σ_j p_kj²/b_j
    term_c = (p * p / (a + _EPS)).sum(axis=0, keepdims=True) / (b**2 + _EPS)  # (1/b_l^2) Σ_i p_il²/a_i
    grad_phi2 = 2.0 * p / (e + _EPS) - term_r - term_c

    grad_v = grad_phi2 / (2.0 * (q - 1) * V)
    p_flat = p.ravel()
    sigma = (np.diag(p_flat) - np.outer(p_flat, p_flat)) / n
    var_v = float(grad_v.ravel() @ sigma @ grad_v.ravel())
    return V, var_v


def cramers_equal_p(V0: float, var0: float, V1: float, var1: float) -> float:
    """Two-sided z-test p-value for H0: V0 = V1 (delta-method variances)."""
    denom = np.sqrt(var0 + var1)
    if denom == 0 or np.isnan(denom):
        return np.nan
    z = (V0 - V1) / denom
    return float(2 * norm.sf(abs(z)))


def equal_assoc_cat_cat_codes(
    x0: np.ndarray, y0: np.ndarray,
    x1: np.ndarray, y1: np.ndarray,
    nr: int, nc: int,
) -> float:
    """Fast port of ``equal_assoc_cat_cat`` on pre-encoded integer codes.

    Codes were built over the union of categories in the two FULL frames; here
    the tables are sliced to the categories observed in the current pair of
    bootstrap samples (union across samples), matching the original's
    per-iteration crosstab over sampled categories, including the q = min(r, c)
    used by Cramér's V and the effective-dimension >= 2 guards.
    """
    ct0 = bincount2(x0, y0, nr, nc)
    ct1 = bincount2(x1, y1, nr, nc)
    obs_r = (ct0.sum(axis=1) + ct1.sum(axis=1)) > 0
    obs_c = (ct0.sum(axis=0) + ct1.sum(axis=0)) > 0
    ct0 = ct0[obs_r][:, obs_c]
    ct1 = ct1[obs_r][:, obs_c]
    if (ct0.sum(axis=1) > 0).sum() < 2 or (ct0.sum(axis=0) > 0).sum() < 2 \
            or (ct1.sum(axis=1) > 0).sum() < 2 or (ct1.sum(axis=0) > 0).sum() < 2:
        return np.nan
    V0, var0 = cramers_v_from_counts(ct0)
    V1, var1 = cramers_v_from_counts(ct1)
    if np.isnan(V0) or np.isnan(V1):
        return np.nan
    return cramers_equal_p(V0, var0, V1, var1)


def eta_squared_codes(num: np.ndarray, cat: np.ndarray, k: int) -> tuple[float, float]:
    """``_eta_sq`` on pre-encoded codes: closed-form, zero-count groups ignored."""
    n = len(num)
    if n < 5:
        return np.nan, np.nan
    n_g = np.bincount(cat, minlength=k).astype(float)
    k_obs = int((n_g > 0).sum())
    if k_obs < 2:
        return np.nan, np.nan
    sum_g = np.bincount(cat, weights=num, minlength=k)
    grand = num.mean()
    means_g = np.where(n_g > 0, sum_g / np.where(n_g > 0, n_g, 1.0), grand)
    ssb = float((n_g * (means_g - grand) ** 2).sum())
    sst = float(((num - grand) ** 2).sum())
    if sst <= 0:
        return np.nan, np.nan
    eta2 = ssb / sst
    return eta2, (4 * eta2 * (1 - eta2) ** 2) / n


def equal_assoc_num_cat_codes(
    num0: np.ndarray, cat0: np.ndarray,
    num1: np.ndarray, cat1: np.ndarray,
    k: int,
) -> float:
    """Fast port of ``equal_assoc_num_cat`` on pre-encoded codes."""
    eta0, var0 = eta_squared_codes(num0, cat0, k)
    eta1, var1 = eta_squared_codes(num1, cat1, k)
    if np.isnan(eta0) or np.isnan(eta1) or eta0 == eta1:
        return np.nan
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        z = (eta0 - eta1) / np.sqrt(var0 + var1)
    return float(2 * norm.sf(abs(z)))


def eta_squared(x: np.ndarray, codes: np.ndarray, k: int) -> tuple[float, float, int]:
    """One-way eta-squared via group means (identical to OLS SS_between/SS_total).

    Returns (eta2, var_eta2, n) with the same delta-method variance
    ``4*eta2*(1-eta2)^2/n`` used by SSDataBench.
    """
    x = np.asarray(x, dtype=float)
    codes = np.asarray(codes, dtype=np.int64)
    n = len(x)
    if n < 5 or k < 2:
        return np.nan, np.nan, n
    n_g = np.bincount(codes, minlength=k).astype(float)
    sum_g = np.bincount(codes, weights=x, minlength=k)
    grand = x.mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        means_g = np.where(n_g > 0, sum_g / np.where(n_g > 0, n_g, 1.0), grand)
    ssb = float((n_g * (means_g - grand) ** 2).sum())
    sst = float(((x - grand) ** 2).sum())
    if sst <= 0:
        return np.nan, np.nan, n
    eta2 = ssb / sst
    return eta2, (4 * eta2 * (1 - eta2) ** 2) / n, n


def corr_fisher_p(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray) -> float:
    """Fisher-z test p-value for H0: corr1 = corr2."""
    r1 = np.corrcoef(x1, y1)[0, 1]
    r2 = np.corrcoef(x2, y2)[0, 1]
    if np.isnan(r1) or np.isnan(r2):
        return np.nan
    r1, r2 = np.clip(r1, -0.999999, 0.999999), np.clip(r2, -0.999999, 0.999999)
    se = np.sqrt(1 / (len(x1) - 3) + 1 / (len(x2) - 3))
    z = (np.arctanh(r1) - np.arctanh(r2)) / se
    return float(2 * norm.sf(abs(z)))


def encode_categories(*series) -> list[np.ndarray]:
    """Encode several series onto their sorted union of categories as int codes."""
    import pandas as pd

    cats = sorted(set().union(*(set(pd.Series(s).dropna()) for s in series)))
    out = []
    for s in series:
        codes = pd.Categorical(pd.Series(s), categories=cats).codes
        out.append(np.asarray(codes, dtype=np.int64))
    return out
