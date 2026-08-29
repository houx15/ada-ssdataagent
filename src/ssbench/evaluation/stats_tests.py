"""Delta-method association-strength tests (port of SSDataBench type2/type5 helpers)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def _cramers_v_and_var(tab) -> tuple[float, float]:
    """Cramér's V and its multinomial delta-method variance via autograd jacobian."""
    try:
        import autograd.numpy as anp
        from autograd import jacobian
    except ImportError as e:  # pragma: no cover
        raise ImportError("Please first install autograd: pip install autograd") from e

    cm = np.array(tab, dtype=float)
    n = np.sum(cm)
    r, c = cm.shape
    q = min(r, c)
    if n == 0 or q <= 1:
        return np.nan, np.nan
    p_hat = (cm / n).flatten()

    def cramers_v_func(p_vec):
        p_mat = p_vec.reshape((r, c))
        p_row = anp.sum(p_mat, axis=1, keepdims=True)
        p_col = anp.sum(p_mat, axis=0, keepdims=True)
        expected = anp.dot(p_row, p_col)
        term = (p_mat - expected) ** 2 / (expected + 1e-20)
        phi2 = anp.sum(term)
        return anp.sqrt(phi2 / (q - 1))

    try:
        V_est = cramers_v_func(p_hat)
        if V_est < 1e-6:
            return 0.0, 0.0
        J = jacobian(cramers_v_func)(p_hat)
        Sigma = (np.diag(p_hat) - np.outer(p_hat, p_hat)) / n
        var_V = np.dot(J, np.dot(Sigma, J.T))
        return float(V_est), float(var_V)
    except Exception:  # noqa: BLE001
        return np.nan, np.nan


def equal_assoc_cat_cat(df: pd.DataFrame, v1: str, v2: str, group_col="__grp__") -> float:
    """z-test of H0: V_real = V_sim for a categorical pair."""
    groups = df[group_col].unique()
    if len(groups) != 2:
        return np.nan
    g0, g1 = sorted(groups)
    df0 = df[df[group_col] == g0]
    df1 = df[df[group_col] == g1]

    cats_v1 = sorted(set(df0[v1].dropna()) | set(df1[v1].dropna()))
    cats_v2 = sorted(set(df0[v2].dropna()) | set(df1[v2].dropna()))

    x0 = pd.Categorical(df0[v1], categories=cats_v1)
    y0 = pd.Categorical(df0[v2], categories=cats_v2)
    x1 = pd.Categorical(df1[v1], categories=cats_v1)
    y1 = pd.Categorical(df1[v2], categories=cats_v2)

    ct0 = pd.crosstab(x0, y0).reindex(index=cats_v1, columns=cats_v2, fill_value=0)
    ct1 = pd.crosstab(x1, y1).reindex(index=cats_v1, columns=cats_v2, fill_value=0)

    def _eff_dim(ct):
        return (ct.sum(axis=1) > 0).sum(), (ct.sum(axis=0) > 0).sum()

    (r0, c0), (r1, c1) = _eff_dim(ct0), _eff_dim(ct1)
    if r0 < 2 or c0 < 2 or r1 < 2 or c1 < 2:
        return np.nan

    V0, var0 = _cramers_v_and_var(ct0)
    V1, var1 = _cramers_v_and_var(ct1)
    if np.isnan(V0) or np.isnan(V1):
        return np.nan

    z = (V0 - V1) / np.sqrt(var0 + var1)
    return float(2 * norm.sf(abs(z)))


def equal_assoc_num_num(x1, y1, x2, y2) -> float:
    """Fisher-z test of H0: corr_real = corr_sim."""
    r1 = np.corrcoef(x1, y1)[0, 1]
    r2 = np.corrcoef(x2, y2)[0, 1]
    if np.isnan(r1) or np.isnan(r2):
        return np.nan
    r1, r2 = np.clip(r1, -0.999999, 0.999999), np.clip(r2, -0.999999, 0.999999)
    z1, z2 = np.arctanh(r1), np.arctanh(r2)
    se = np.sqrt(1 / (len(x1) - 3) + 1 / (len(x2) - 3))
    z = (z1 - z2) / se
    return float(2 * norm.sf(abs(z)))


def _eta_sq(num, cat) -> tuple[float, float, int]:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    df = pd.DataFrame({"num": num, "cat": cat}).dropna()
    if df["cat"].nunique() < 2 or len(df) < 5:
        return np.nan, np.nan, np.nan
    try:
        model = smf.ols("num ~ C(cat)", data=df).fit()
        anova = sm.stats.anova_lm(model, typ=2)
        ss_between = anova.loc["C(cat)", "sum_sq"]
        ss_total = ss_between + anova.loc["Residual", "sum_sq"]
        eta2 = ss_between / ss_total
        n = len(df)
        var_eta2 = (4 * eta2 * (1 - eta2) ** 2) / n
        return eta2, var_eta2, n
    except Exception:  # noqa: BLE001
        return np.nan, np.nan, np.nan


def equal_assoc_num_cat(num1, cat1, num2, cat2) -> float:
    """z-test of H0: eta2_real = eta2_sim (numeric ~ categorical)."""
    eta1, var1, _ = _eta_sq(num1, cat1)
    eta2, var2, _ = _eta_sq(num2, cat2)
    if np.isnan(eta1) or np.isnan(eta2) or eta1 == eta2:
        return np.nan
    z = (eta1 - eta2) / np.sqrt(var1 + var2)
    return float(2 * norm.sf(abs(z)))


def equal_r2(r2_1: float, n1: int, r2_2: float, n2: int) -> float:
    """z-test of H0: R2_real = R2_sim via 4*R2*(1-R2)^2/n variances."""
    se1 = np.sqrt(4 * r2_1 * (1 - r2_1) ** 2 / n1)
    se2 = np.sqrt(4 * r2_2 * (1 - r2_2) ** 2 / n2)
    z = (r2_1 - r2_2) / np.sqrt(se1**2 + se2**2)
    return float(2 * norm.sf(abs(z)))
