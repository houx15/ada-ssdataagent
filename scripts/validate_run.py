#!/usr/bin/env python3
"""Write schema, cross-field, and longitudinal validity diagnostics for a run."""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ssbench.datasets.registry import load_dataset  # noqa: E402
from ssbench.diagnostics.consistency import validate_table  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    with open(os.path.join(args.run_dir, "meta.json"), encoding="utf-8") as handle:
        meta = json.load(handle)
    df = pd.read_csv(os.path.join(args.run_dir, "sim.csv"), low_memory=False)
    rows = [r.to_dict() for r in validate_table(df, load_dataset(meta["dataset"]))]
    out = args.out_dir or os.path.join(args.run_dir, "validation")
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "consistency.csv"), index=False)
    summary = {
        "dataset": meta["dataset"],
        "run_dir": os.path.abspath(args.run_dir),
        "n_rows": len(df),
        "n_rules": len(rows),
        "rules_with_violations": sum(r["violations"] > 0 for r in rows),
        "results": rows,
    }
    with open(os.path.join(out, "consistency.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"validation -> {out} ({summary['rules_with_violations']}/{len(rows)} rules violated)")


if __name__ == "__main__":
    main()
