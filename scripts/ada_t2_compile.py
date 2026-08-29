#!/usr/bin/env python3
"""Compile T2 probe JSONL into the loader's explicit pair/rho target format."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from ada_t2_load import pooled_targets  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    targets = pooled_targets(args.probe)
    rows = [
        {"pair": list(pair), "rho": rho}
        for pair, rho in sorted(targets.items())
    ]
    if not rows:
        raise SystemExit("no valid T2 targets compiled")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
    print(f"written {args.out}: {len(rows)} pairs")


if __name__ == "__main__":
    main()
