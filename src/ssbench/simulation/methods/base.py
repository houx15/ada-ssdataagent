"""Method interface shared by all generation paradigms."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from ssbench.datasets.schema import DatasetSpec

_REGISTRY: dict[str, type] = {}


@runtime_checkable
class GenerationMethod(Protocol):
    """A method generates one synthetic row per conditioning input row."""

    name: str

    def generate(
        self,
        spec: DatasetSpec,
        inputs_df: pd.DataFrame,
        failure_logger=None,
    ) -> pd.DataFrame: ...


def register_method(cls):
    _REGISTRY[cls.name] = cls
    return cls


def create_method(name: str, **kwargs) -> GenerationMethod:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown method '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
