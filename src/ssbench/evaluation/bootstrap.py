"""Bootstrap scaffolding shared by T1–T5 (matched draws from real & sim)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BootstrapConfig:
    B: int = 100
    alpha: float = 0.05
    sample_n: int = 500
    id_col: str = "profile_id"
    seed: int | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "BootstrapConfig":
        return cls(
            B=int(d.get("B", 100)),
            alpha=float(d.get("alpha", 0.05)),
            sample_n=int(d.get("sample_n", 500)),
            seed=(int(d["seed"]) if d.get("seed") is not None else None),
        )


class RngPair:
    """Fast matched bootstrap index draws (numpy Generator, no per-call seeding)."""

    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)

    def indices(self, n_real: int, n_sim: int, sample_n: int) -> tuple[np.ndarray, np.ndarray]:
        ri = self._rng.integers(0, n_real, size=sample_n)
        si = self._rng.integers(0, n_sim, size=sample_n)
        return ri, si

    def draw(self, df_real: pd.DataFrame, df_sim: pd.DataFrame, sample_n: int):
        ri, si = self.indices(len(df_real), len(df_sim), sample_n)
        return df_real.iloc[ri], df_sim.iloc[si]
