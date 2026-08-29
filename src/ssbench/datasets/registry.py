"""Registry of known datasets, backed by configs/datasets/*.yaml."""

from __future__ import annotations

import os

import yaml

from ssbench.datasets.schema import DatasetSpec
from ssbench.settings import get_settings


def available_datasets() -> list[str]:
    cfg_dir = os.path.join(get_settings().configs_dir, "datasets")
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(cfg_dir)
        if f.endswith((".yaml", ".yml"))
    )


def load_dataset(name: str) -> DatasetSpec:
    path = os.path.join(get_settings().configs_dir, "datasets", f"{name}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Unknown dataset '{name}'. Available: {available_datasets()}"
        )
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return DatasetSpec.from_yaml(name, raw)
