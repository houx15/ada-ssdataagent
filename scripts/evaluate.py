#!/usr/bin/env python3
"""Evaluate a simulation run directory (T1–T5 per dataset config)."""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "src")

from ssbench.evaluation.runner import evaluate_run


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--types", default=None, help="comma list, e.g. t1,t2 (default: dataset config)")
    ap.add_argument("--B", type=int, default=None)
    ap.add_argument("--sample-n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None,
                    help="fixed evaluator bootstrap seed")
    ap.add_argument("--output-dir", default=None,
                    help="separate output directory for this evaluator repeat")
    args = ap.parse_args()

    types = args.types.split(",") if args.types else None
    evaluate_run(args.run_dir, types=types, B=args.B, sample_n=args.sample_n,
                 seed=args.seed, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
