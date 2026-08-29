#!/usr/bin/env python3
"""Prepare processed reference samples for one or all datasets."""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "src")

from ssbench.datasets.registry import available_datasets, load_dataset
from ssbench.preprocessing.prepare import prepare_dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="all", help=f"{available_datasets()} or all")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    names = available_datasets() if args.dataset == "all" else [args.dataset]
    for name in names:
        prepare_dataset(load_dataset(name), n=args.n, seed=args.seed)


if __name__ == "__main__":
    main()
