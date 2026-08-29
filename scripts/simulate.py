#!/usr/bin/env python3
"""Run a simulation for a dataset with a chosen generation method."""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "src")

from ssbench.simulation.runner import run_simulation


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--method", default="direct")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--model", default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--max-attempts", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--resume-dir", default=None, help="existing run directory to continue")
    ap.add_argument("--run-dir", default=None,
                    help="exact output directory for a new run")
    args = ap.parse_args()

    run_simulation(
        dataset=args.dataset,
        method=args.method,
        n=args.n,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        max_attempts=args.max_attempts,
        seed=args.seed,
        tag=args.tag,
        resume_dir=args.resume_dir,
        run_dir=args.run_dir,
    )


if __name__ == "__main__":
    main()
