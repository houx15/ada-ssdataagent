"""Dataset specification loading (configs/datasets/*.yaml)."""

from ssbench.datasets.registry import available_datasets, load_dataset

__all__ = ["load_dataset", "available_datasets"]
