"""Edge aggregation, Hodge projection, and ADA-innovation analysis (v2 §8)."""

from __future__ import annotations

import numpy as np


def aggregate_edges(results: list, cats: list[str]):
    """Mean antisymmetric edge response per (j,k), plus per-source means."""
    m = len(cats)
    acc = {(j, k): [] for j in range(m) for k in range(j + 1, m)}
    src_acc = {}
    for r in results:
        for e in r.edges:
            key = e["edge"]
            if key in r.g:
                acc[key].append(r.g[key])
                src_acc.setdefault((key, e["source"]), []).append(r.g[key])
    gbar = {k: float(np.mean(v)) for k, v in acc.items() if v}
    counts = {k: len(v) for k, v in acc.items()}
    src_g = {f"{k[0]}_{k[1]}_{s}": float(np.mean(v)) for (k, s), v in src_acc.items() if v}
    return gbar, counts, src_g


def hodge(gbar: dict, cats: list[str], lam: float = 0.0):
    """Weighted least-squares node potential, zero-sum gauge.

    Solves min ||W^{1/2}(B phi - gbar)||^2 (+ lam * phi' L_phi phi),
    where (B phi)_{j->k} = phi_k - phi_j.
    """
    m = len(cats)
    L = np.zeros((m, m))
    b = np.zeros(m)
    W = 1.0
    for (j, k), g in gbar.items():
        L[j, j] += W
        L[k, k] += W
        L[j, k] -= W
        L[k, j] -= W
        b[k] += W * g
        b[j] -= W * g
    A = L + lam * np.eye(m) + 1e-9 * np.ones((m, m))
    try:
        phi = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        phi = np.linalg.pinv(A) @ b
    phi -= phi.mean()
    resid = {e: g - (phi[e[1]] - phi[e[0]]) for e, g in gbar.items()}
    return phi, resid


def clr_with_pseudo(counts: np.ndarray, delta: float = 0.5) -> np.ndarray:
    p = (counts + delta) / (counts.sum() + delta * len(counts))
    x = np.log(p)
    return x - x.mean()


def softmax(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max())
    return z / z.sum()
